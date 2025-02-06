#ifndef EVALUATOR_H
#define EVALUATOR_H

#include "contour_db.h"
#include "contour_mng.h"
#include "tools/pointcloud_util.h"
#include "tools/algos.h"

template <int dim>
struct SimpleRMSE
{
    double sum_sqs = 0; // 误差和
    double sum_abs = 0; // 误差开发和
    int cnt_sqs = 0;

    SimpleRMSE() = default;

    // 添加一个误差
    void addOneErr(const double *d)
    {
        cnt_sqs++;
        double tmp = 0;
        for (int i = 0; i < dim; i++)
        {
            tmp += d[i] * d[i];
        }
        sum_sqs += tmp;
        sum_abs += std::sqrt(tmp);
    }

    double getRMSE() const { return cnt_sqs ? std::sqrt(sum_sqs / cnt_sqs) : -1; }

    double getMean() const { return cnt_sqs ? sum_abs / cnt_sqs : -1; }
};

// 输出的预测值结构体，包括source-loop id TFNP 相关度 还有pose error
struct PredictionOutcome
{
    enum Res
    {
        TP,
        FP,
        TN,
        FN
    };

    int id_src = -1; //
    int id_tgt = -1;
    Res tfpn = Res::TN;  // the most insignificant type
    double est_err[3]{}; // TP, FP: the error param on SE2, else: all zero
    double correlation{};

    // 跑数据用
    Eigen::Isometry3d match_pose = Eigen::Isometry3d::Identity();
    std::string match_lidar_path = "";

    // xwl
    std::vector<std::pair<int, int>> data_size; // 各层的cont_views_的size和
    std::pair<int, int> all_cont_data_size;     // cont_views_的size总和
};

struct LaserScanInfo
{                                    // 点云帧基本参数
    bool has_gt_positive_lc = false; // 回环帧标志
    Eigen::Isometry3d sens_pose;     // gt pose
    int seq = 0;                     // 序列号
    double ts = 0;                   // 时间戳
    std::string fpath;               // 保存地址
    LaserScanInfo() = default;
};

// Definition of loader
// 1. use script to generate 2 files: (all ts are in ns)
//  1) timestamp and gt pose (z up) of the sensor. Ordered by gt ts. (13 elements per line)
//  2) timestamp, seq, and the path (no space) of each lidar scan bin file.
//    Ordered by lidar ts AND seq. (3 elements per line)
// 2. load the .bin data in sequence, and find the gt pose
class ContLCDEvaluator
{

    // valid input info about lidar scans
    std::vector<LaserScanInfo> laser_info_; // use predefined {seq-id, bin file} for better traceability //保存pose 点云地址 时间戳
    std::vector<int> assigned_seqs_;        // actually a seq:addr map  //保存点云序号

    // param:
    const double ts_diff_tol = 10e-3;  // 10ms. In Mulran, the gt is given at an interval of about 10ms
    const double min_time_excl = 15.0; // exclude 15s
    const double sim_thres;            // the similarity score when determining TFPN  相似性得分？还是阈值？

    // bookkeeping variables
    int p_lidar_curr = -1;
    //  std::map<int, LaserScanInfo>::iterator it_lidar_curr;  // int index is a little safer than iterator/pointer?

    // benchmark recorders
    SimpleRMSE<2> tp_trans_rmse, all_trans_rmse; // tp的rmse
    SimpleRMSE<1> tp_rot_rmse, all_rot_rmse;     // 所有的rmse
    std::vector<PredictionOutcome> pred_records; // 每一帧处理之后的储存容器

public:
    ContLCDEvaluator(const double &bar) : sim_thres(bar)
    {
    }

    bool loadNewScan()
    {
        p_lidar_curr++;
        // Load the scan into the cache so that it can be retrieved by calling related functions
        CHECK(p_lidar_curr >= 0);
        // printf("p_lidar_curr: %d\n", p_lidar_curr);
        // printf("laser_info_ size: %d\n", laser_info_.size());
        // if (p_lidar_curr >= laser_info_.size()) {
        //   printf("\n===\ncurrent addr %d exceeds boundary\n", p_lidar_curr);
        //   return false;
        // }

        // printf("\n===\nloaded scan addr %d, seq: %d, fpath: %s\n", p_lidar_curr, laser_info_[p_lidar_curr].seq,
        //  laser_info_[p_lidar_curr].fpath.c_str());
        return true;
    }

    // 1. loader
    // 获取当前点云帧数据结构体 LaserScanInfo
    const LaserScanInfo &getCurrScanInfo() const
    {
        CHECK(p_lidar_curr < laser_info_.size());
        CHECK(p_lidar_curr >= 0);

        return laser_info_[p_lidar_curr];
    }

    // 填入当前点云帧数据结构体
    void pushCurrScanInfo(LaserScanInfo &info_)
    {
        // 寻找真值回环帧
        int cnt_gt_lc_p = 0;
        int cnt_gt_lc = 0;
        for (auto &it_slow : laser_info_)
        {
            if (info_.ts < it_slow.ts + min_time_excl)
                break;
            double dist = (info_.sens_pose.translation() - it_slow.sens_pose.translation()).norm();
            if (dist < 5.0)
            {
                if (!info_.has_gt_positive_lc)
                {
                    info_.has_gt_positive_lc = true;
                    cnt_gt_lc_p++;
                }
                cnt_gt_lc++;
            }
        }   

        laser_info_.emplace_back(info_);
    }

    // 返回帧数
    //  const size_t &getScanInfoNum() const {
    //    return laser_info_.size();
    //  }

    // 获取评价数据
    const PredictionOutcome &getCurrEvaData() const
    {
        CHECK(p_lidar_curr < laser_info_.size());
        CHECK(p_lidar_curr >= 0);

        return pred_records[p_lidar_curr];
    }

    // 建立并返回一个抽象结构 的结构体指针
    std::shared_ptr<ContourManager> getCurrContourManager(const ContourManagerConfig &config) const
    {
        // assumption:
        // 1. The returned cont_mng is matched to the data in `laser_info_`
        // 2. The int index of every cont_mng is matched to the index of every item of `laser_info`
        //   (so that we can use cont_mng as input to index gt 3d poses in the "recorder" below)

        std::shared_ptr<ContourManager> cmng_ptr(new ContourManager(config, laser_info_[p_lidar_curr].seq));
        pcl::PointCloud<pcl::PointXYZ>::ConstPtr out_ptr = readKITTIPointCloudBin<pcl::PointXYZ>(
            laser_info_[p_lidar_curr].fpath); // 读取对应bin中的点云

        std::string str_id = std::to_string(laser_info_[p_lidar_curr].seq);       // 读取帧序号并转换成string
        str_id = "assigned_id_" + std::string(8 - str_id.length(), '0') + str_id; // 拼接

        // cmng_ptr->makeBEV<pcl::PointXYZ>(out_ptr, str_id);      //制作bev并保存
        // cmng_ptr->makeContoursRecurs();                         //contour 以每一层中每一个anchor为单元 生成key和bci并保存
        return cmng_ptr;
    }

    // 2. recorder.
    // 求解查询帧的prediction outcome
    /// \param q_mng 查询帧的描述符指针
    /// \param est_corr 位姿估计误差
    /// \param cand_mng 匹配帧的描述符指针
    /// \param T_est_delta_2d 最终的候选位姿
    PredictionOutcome
    addPrediction(const std::shared_ptr<const ContourManager> &q_mng, double est_corr,
                  const std::shared_ptr<const ContourManager> &cand_mng = nullptr,
                  const Eigen::Isometry2d &T_est_delta_2d = Eigen::Isometry2d::Identity())
    {
        int id_tgt = q_mng->getIntID(); //// q: src, cand: tgt 这个注释有问题吧
        int addr_tgt = id_tgt;
        CHECK_GE(addr_tgt, 0);

        PredictionOutcome curr_res;
        curr_res.id_tgt = id_tgt;
        curr_res.correlation = est_corr;
        int addr_src_ = -1;

        if (cand_mng)
        {
            // The prediction is positive
            int id_src = cand_mng->getIntID();
            int addr_src = id_src;
            CHECK_GE(addr_src, 0);
            addr_src_ = addr_src;

            curr_res.id_src = id_src;

            const auto gen_bev_config = q_mng->getConfig();                                                                // the config used to generate BEV
            Eigen::Isometry2d tf_err = ConstellCorrelation::evalMetricEst(T_est_delta_2d, laser_info_[addr_src].sens_pose, //! 位姿误差在这里求解
                                                                          laser_info_[addr_tgt].sens_pose, gen_bev_config);
            // TODO 考虑加入时间连续性？ 如何辅助提高位姿？
            double est_trans_norm2d = ConstellCorrelation::getEstSensTF(T_est_delta_2d, gen_bev_config).translation().norm();
            double gt_trans_norm3d = (laser_info_[addr_src].sens_pose.translation() -
                                      laser_info_[addr_tgt].sens_pose.translation())
                                         .norm();
            printf(" Dist: Est2d: %.2f; GT3d: %.2f\n", est_trans_norm2d, gt_trans_norm3d);

            double err_vec[3] = {tf_err.translation().x(), tf_err.translation().y(), std::atan2(tf_err(1, 0), tf_err(0, 0))};
            printf(" Error: dx=%f, dy=%f, dtheta=%f\n", err_vec[0], err_vec[1], err_vec[2]);

            memcpy(curr_res.est_err, err_vec, sizeof(err_vec)); // 保存
            if (est_corr >= sim_thres)
            {
                if (laser_info_[addr_tgt].has_gt_positive_lc && gt_trans_norm3d < 5.0)
                { // TP   //这里是真值标准
                    curr_res.tfpn = PredictionOutcome::TP;

                    tp_trans_rmse.addOneErr(err_vec);
                    tp_rot_rmse.addOneErr(err_vec + 2);
                }
                else
                { // FP
                    curr_res.tfpn = PredictionOutcome::FP;
                }
            }
            else
            {
                if (laser_info_[addr_tgt].has_gt_positive_lc)
                    curr_res.tfpn = PredictionOutcome::FN;
                else
                    curr_res.tfpn = PredictionOutcome::TN;
            }

            all_trans_rmse.addOneErr(err_vec);
            all_rot_rmse.addOneErr(err_vec + 2);
        }
        else
        {
            // The prediction is negative
            if (laser_info_[addr_tgt].has_gt_positive_lc) // FN
                curr_res.tfpn = PredictionOutcome::FN;    // gt loop 但判断为不回环
            else                                          // TN
                curr_res.tfpn = PredictionOutcome::TN;    // gt not loop est not loop
        }

        // xwl cont_views_ 数据大小计算
        auto q_config_ = q_mng->getConfig();
        curr_res.all_cont_data_size = {0, 0};
        for (int i = 0; i < q_config_.lv_grads_.size(); i++)
        {
            auto q_levelcont_ = q_mng->getLevContours(i);
            if (q_levelcont_.empty())
                break;
            int level_cont_size = (q_levelcont_.size()) * q_mng->getSize(i);
            int level_cont_size_test = q_levelcont_.size() * q_mng->getTestSize(i);
            curr_res.data_size.emplace_back(std::pair<int, int>{level_cont_size, level_cont_size_test});
            curr_res.all_cont_data_size.first += level_cont_size;
            curr_res.all_cont_data_size.second += level_cont_size_test;
        }
        // printf(" Datasize: %d Test datasize: %d\n", curr_res.all_cont_data_size.first, curr_res.all_cont_data_size.second);

        // 跑数据处理
        if (curr_res.tfpn == PredictionOutcome::TP || curr_res.tfpn == PredictionOutcome::FP)
        {
            curr_res.match_pose = laser_info_[addr_src_].sens_pose;
            curr_res.match_lidar_path = laser_info_[addr_src_].fpath;
        }

        pred_records.emplace_back(curr_res); // 保存到pred_records
        return curr_res;
    }

    // 3. Result saver
    // 最后的数据处理保存及打印函数
    void savePredictionResults(const std::string &sav_path) const
    {
        std::fstream res_file(sav_path, std::ios::out);

        if (res_file.rdstate() != std::ifstream::goodbit)
        {
            std::cerr << "Error opening " << sav_path << std::endl;
            return;
        }

        // tgt before src
        for (const auto &rec : pred_records)
        {
            int addr_tgt = lookupNN<int>(rec.id_tgt, assigned_seqs_, 0);
            CHECK_GE(addr_tgt, 0);

            res_file << rec.tfpn << "\t";

            std::string str_rep_tgt = laser_info_[addr_tgt].fpath, str_rep_src;

            if (rec.id_src < 0)
            {
                res_file << rec.id_tgt << "-x" << "\t";
                str_rep_src = "x";
            }
            else
            {
                int addr_src = lookupNN<int>(rec.id_src, assigned_seqs_, 0);
                CHECK_GE(addr_src, 0);

                res_file << rec.id_tgt << "-" << rec.id_src << "\t";
                str_rep_src = laser_info_[addr_src].fpath;
            }

            res_file << rec.correlation << "\t" << rec.est_err[0] << "\t" << rec.est_err[1] << "\t" << rec.est_err[2] << "\t";

            //      // case 1: path
            //      res_file << str_rep_tgt << "\t" << str_rep_src << "\n"; // may be too long

            // case 2: shortened
            int str_max_len = 32;
            int beg_tgt = std::max(0, (int)str_rep_tgt.length() - str_max_len);
            int beg_src = std::max(0, (int)str_rep_src.length() - str_max_len);
            res_file << str_rep_tgt.substr(beg_tgt, str_rep_tgt.length() - beg_tgt) << "\t"
                     << str_rep_src.substr(beg_src, str_rep_src.length() - beg_src) << "\n";
        }
        // rmse and mean error can be calculated from this file. So we will not record it.

        printf("In outcome file:\n");
        printf("TP is %d\n", static_cast<std::underlying_type<PredictionOutcome::Res>::type>(PredictionOutcome::TP));
        printf("FP is %d\n", static_cast<std::underlying_type<PredictionOutcome::Res>::type>(PredictionOutcome::FP));
        printf("TN is %d\n", static_cast<std::underlying_type<PredictionOutcome::Res>::type>(PredictionOutcome::TN));
        printf("FN is %d\n", static_cast<std::underlying_type<PredictionOutcome::Res>::type>(PredictionOutcome::FN));

        res_file.close();
        printf("Outcome saved successfully.\n");

        // 打印数据总量
        int dataset_cont_size = 0;
        int dataset_cont_size_test = 0;
        for (const auto &rec : pred_records)
        {
            dataset_cont_size += rec.all_cont_data_size.first;
            dataset_cont_size_test += rec.all_cont_data_size.second;
        }
        printf(" Datasize: %d Test datasize: %d\n", dataset_cont_size, dataset_cont_size_test);
    }

    inline double getTPMeanTrans() const { return tp_trans_rmse.getMean(); }

    inline double getTPMeanRot() const { return tp_rot_rmse.getMean(); }

    inline double getTPRMSETrans() const { return tp_trans_rmse.getRMSE(); }

    inline double getTPRMSERot() const { return tp_rot_rmse.getRMSE(); }

    // 4. related public util
    static void loadCheckThres(const std::string &fpath, CandidateScoreEnsemble &thres_lb,
                               CandidateScoreEnsemble &thres_ub);
};

#endif
