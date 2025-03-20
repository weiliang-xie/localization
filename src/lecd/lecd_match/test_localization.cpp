#include <memory>
#include <utility>

#include "contour_db.h"
#include "evaluator.h"
#include "cont2_ros/spinner_ros.h"
#include "tools/bm_util.h"
#include "tools/config_handler.h"
#include "lidar_rec.h"

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <mutex>
// #include <matplotlibcpp.h>

std::mutex mtx;  // 互斥锁

// namespace plt = matplotlibcpp;

const std::string PROJ_DIR = std::string(PJSRCDIR);

SequentialTimeProfiler stp;

std::unique_ptr<ContourDB> ptr_contour_db;       // 描述符数据库指针
std::unique_ptr<ContLCDEvaluator> ptr_evaluator; // 评价部分指针

ContourDBConfig db_config;
ContourManagerConfig cm_config;
CandidateScoreEnsemble thres_lb_, thres_ub_; // check thresholds variable query parameters 查询阈值 分上界下界

std::string fpath_outcome_sav;
pcl::PointXYZRGB vec2point(const Eigen::Vector3d &vec, std::vector<std::uint8_t> &rgb)
{
    pcl::PointXYZRGB pi;
    pi.x = vec[0];
    pi.y = vec[1];
    pi.z = vec[2];
    pi.r = rgb[0]; // 红色
    pi.g = rgb[1]; // 绿色
    pi.b = rgb[2]; // 蓝色
    return pi;
}
Eigen::Vector3d point2vec(const pcl::PointXYZRGB &pi)
{
    return Eigen::Vector3d(pi.x, pi.y, pi.z);
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr readCloud(std::string lidar_path, Eigen::Isometry3d pose, std::vector<std::uint8_t> rgb)
{
    // Read KITTI data
    std::ifstream lidar_data_file;
    std::vector<float> lidar_data_buffer = {};
    lidar_data_file.open(lidar_path,
                         std::ifstream::in | std::ifstream::binary);
    if (!lidar_data_file)
    {
        std::cout << "Read End..." << std::endl;
        return nullptr;
    }
    else
    {
        lidar_data_file.seekg(0, std::ios::end);
        const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
        lidar_data_file.seekg(0, std::ios::beg);

        lidar_data_buffer.resize(num_elements);
        lidar_data_file.read(reinterpret_cast<char *>(&lidar_data_buffer[0]),
                             num_elements * sizeof(float));
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud(
        new pcl::PointCloud<pcl::PointXYZRGB>());
    for (std::size_t i = 0; i < lidar_data_buffer.size(); i += 4)
    {
        pcl::PointXYZRGB point;
        point.x = lidar_data_buffer[i];
        point.y = lidar_data_buffer[i + 1];
        point.z = lidar_data_buffer[i + 2];
        Eigen::Vector3d pv = point2vec(point);
        pv = pose.rotation() * pv + pose.translation();
        point = vec2point(pv, rgb);
        temp_cloud->push_back(point);
    }
    return temp_cloud;
}

void loadConfig(const std::string &config_fpath, std::string &sav_path)
{
    printf("Loading parameters...\n");
    auto yl = yamlLoader(config_fpath);
    std::string fpath_sens_gt_pose, fpath_lidar_bins;
    double corr_thres;
    yl.loadOneConfig({"correlation_thres"}, corr_thres);
    ptr_evaluator = std::make_unique<ContLCDEvaluator>(corr_thres); // 导入阈值

    yl.loadOneConfig({"LECDDBConfig", "nnk_"}, db_config.nnk_);
    yl.loadOneConfig({"LECDDBConfig", "max_fine_opt_"}, db_config.max_fine_opt_);
    yl.loadSeqConfig({"LECDDBConfig", "q_levels_"}, db_config.q_levels_);
    yl.loadOneConfig({"LECDDBConfig", "TreeBucketConfig", "max_elapse_"}, db_config.tb_cfg_.max_elapse_);
    yl.loadOneConfig({"LECDDBConfig", "TreeBucketConfig", "min_elapse_"}, db_config.tb_cfg_.min_elapse_);
    yl.loadOneConfig({"LECDDBConfig", "LECDSimThresConfig", "ta_cell_cnt"}, db_config.cont_sim_cfg_.ta_cell_cnt);
    yl.loadOneConfig({"LECDDBConfig", "LECDSimThresConfig", "tp_cell_cnt"}, db_config.cont_sim_cfg_.tp_cell_cnt);
    yl.loadOneConfig({"LECDDBConfig", "LECDSimThresConfig", "tp_eigval"}, db_config.cont_sim_cfg_.tp_eigval);
    yl.loadOneConfig({"LECDDBConfig", "LECDSimThresConfig", "ta_h_bar"}, db_config.cont_sim_cfg_.ta_h_bar);
    yl.loadOneConfig({"LECDDBConfig", "LECDSimThresConfig", "ta_rcom"}, db_config.cont_sim_cfg_.ta_rcom);
    yl.loadOneConfig({"LECDDBConfig", "LECDSimThresConfig", "tp_rcom"}, db_config.cont_sim_cfg_.tp_rcom);
    ptr_contour_db = std::make_unique<ContourDB>(db_config);

    yl.loadOneConfig({"thres_lb_", "i_ovlp_sum"}, thres_lb_.sim_constell.i_ovlp_sum);
    yl.loadOneConfig({"thres_lb_", "i_ovlp_max_one"}, thres_lb_.sim_constell.i_ovlp_max_one);
    yl.loadOneConfig({"thres_lb_", "i_in_ang_rng"}, thres_lb_.sim_constell.i_in_ang_rng);
    yl.loadOneConfig({"thres_lb_", "i_indiv_sim"}, thres_lb_.sim_pair.i_indiv_sim);
    yl.loadOneConfig({"thres_lb_", "i_orie_sim"}, thres_lb_.sim_pair.i_orie_sim);
    yl.loadOneConfig({"thres_lb_", "correlation"}, thres_lb_.sim_post.correlation);
    yl.loadOneConfig({"thres_lb_", "area_perc"}, thres_lb_.sim_post.area_perc);
    yl.loadOneConfig({"thres_lb_", "neg_est_dist"}, thres_lb_.sim_post.neg_est_dist);
    yl.loadOneConfig({"thres_ub_", "i_ovlp_sum"}, thres_ub_.sim_constell.i_ovlp_sum);
    yl.loadOneConfig({"thres_ub_", "i_ovlp_max_one"}, thres_ub_.sim_constell.i_ovlp_max_one);
    yl.loadOneConfig({"thres_ub_", "i_in_ang_rng"}, thres_ub_.sim_constell.i_in_ang_rng);
    yl.loadOneConfig({"thres_ub_", "i_indiv_sim"}, thres_ub_.sim_pair.i_indiv_sim);
    yl.loadOneConfig({"thres_ub_", "i_orie_sim"}, thres_ub_.sim_pair.i_orie_sim);
    yl.loadOneConfig({"thres_ub_", "correlation"}, thres_ub_.sim_post.correlation);
    yl.loadOneConfig({"thres_ub_", "area_perc"}, thres_ub_.sim_post.area_perc);
    yl.loadOneConfig({"thres_ub_", "neg_est_dist"}, thres_ub_.sim_post.neg_est_dist);
    yl.loadSeqConfig({"LECDManagerConfig", "lv_grads_"}, cm_config.lv_grads_);
    // yl.loadOneConfig({"LECDManagerConfig", "reso_row_"}, cm_config.reso_row_);
    // yl.loadOneConfig({"LECDManagerConfig", "reso_col_"}, cm_config.reso_col_);
    yl.loadOneConfig({"LECDManagerConfig", "n_row_"}, cm_config.n_row_);
    yl.loadOneConfig({"LECDManagerConfig", "n_col_"}, cm_config.n_col_);
    yl.loadOneConfig({"LECDManagerConfig", "lidar_height_"}, cm_config.lidar_height_);
    yl.loadOneConfig({"LECDManagerConfig", "blind_sq_"}, cm_config.blind_sq_);
    yl.loadOneConfig({"LECDManagerConfig", "min_ellipse_key_cnt_"}, cm_config.min_cont_key_cnt_);
    yl.loadOneConfig({"LECDManagerConfig", "min_ellipse_cell_cnt_"}, cm_config.min_cont_cell_cnt_);
    yl.loadOneConfig({"LECDManagerConfig", "piv_firsts_"}, cm_config.piv_firsts_);
    yl.loadOneConfig({"LECDManagerConfig", "dist_firsts_"}, cm_config.dist_firsts_);
    yl.loadOneConfig({"LECDManagerConfig", "roi_radius_"}, cm_config.roi_radius_);
    yl.loadOneConfig({"fpath_outcome_sav"}, sav_path);

    yl.close();
}

std::shared_ptr<ContourManager> LECDtoViews(LECD pt_lecd, const ContourManagerConfig &config, int cur_id)
{
    std::shared_ptr<ContourManager> lecd_ptr(new ContourManager(config, cur_id));
    lecd_ptr->makeContoursRecurs(pt_lecd);
    return lecd_ptr;
}

// 定位线程函数
void localization_thread()
{
    //   ros::NodeHandle nh;
    //   BatchBinSpinner o(nh); // 这个是主要的数据结构 初始化仅传递一个ros句柄
    int cnt = 0;
    int cur_seq = 0;
    int no_data_cnt = 0;

    // TODO 读取地图 待添加
    

    while (true)
    {
        // 判断队列是否为空
        if (lecd_queue.is_empty())
        {               
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); // 模拟处理延迟
            if(lecd_queue.is_empty())
                no_data_cnt++;

            if(no_data_cnt >= 100000)
            {
                ptr_evaluator->savePredictionResults(fpath_outcome_sav);

                //处理时间耗时
                stp.printScreen(true);      //打印最后的耗时统计
                // std::array<double, STAMP_NUM - 1> avg_times;
                // avg_times = pre_times.avgtimes();
                break;
            }
            continue;
        }else{
            no_data_cnt = 0;
        }

        LECD pt_lecd = lecd_queue.pop();

        // CHECK_EQ(pt_lecd.pt_seq, cur_seq);      //检查点云帧序号
        stp.lap(); 

        //转移车端处理时间戳
        // pre_times.pushstamps(pt_lecd.pt_seq, 0, pt_lecd.rec_stamp, pt_lecd.compress_stamp, pt_lecd.tran_stamp);
        stp.compressrecord(pt_lecd.pt_seq);

        // 存储点云信息
        LaserScanInfo info_;
        info_.seq = pt_lecd.pt_seq;
        info_.ts = pt_lecd.time_stamp;
        //gt pose
        Eigen::Vector3d tmp_trans;
        Eigen::Matrix3d tmp_rot_mat;     
        Eigen::Quaterniond tmp_rot_q;
        Eigen::Isometry3d tmp_tf;

        tmp_rot_mat = pt_lecd.gt_pose.block<3,3>(0,0);
        tmp_trans = pt_lecd.gt_pose.block<3,1>(0,3);
        tmp_rot_q = Eigen::Quaterniond(tmp_rot_mat);
        tmp_tf.setIdentity();
        tmp_tf.rotate(tmp_rot_q);
        tmp_tf.pretranslate(tmp_trans); // 完成平移部分和旋转部分处理合并
        info_.sens_pose = tmp_tf;
        info_.fpath = "x";
        
        ptr_evaluator->pushCurrScanInfo(info_);
        ptr_evaluator->loadNewScan();

        std::cout << "current lecd data index: " << ptr_evaluator->getCurrScanInfo().seq << " nums: " << pt_lecd.lecd_nums << std::endl;

        // lecd->lecd_views
        stp.start();
        std::shared_ptr<ContourManager> ptr_cm_tgt = LECDtoViews(pt_lecd, cm_config, cur_seq);
        // 定位模块
        std::vector<std::pair<int, int>> new_lc_pairs;
        std::vector<bool> new_lc_tfp;
        std::vector<std::shared_ptr<const ContourManager>> ptr_cands;
        std::vector<double> cand_corr;
        std::vector<Eigen::Isometry2d> bev_tfs;

        int has_cand_flag = ptr_contour_db->queryRangedKNN(ptr_cm_tgt, thres_lb_, thres_ub_, ptr_cands, cand_corr, bev_tfs); // 查询检索获取候选

        CHECK(ptr_cands.size() < 2);
        PredictionOutcome pred_res;
        if (ptr_cands.empty())
          pred_res = ptr_evaluator->addPrediction(ptr_cm_tgt, 0.0);
        else {
          pred_res = ptr_evaluator->addPrediction(ptr_cm_tgt, cand_corr[0], ptr_cands[0], bev_tfs[0]);  //求解查询帧的prediction outcome
          if (pred_res.tfpn == PredictionOutcome::TP || pred_res.tfpn == PredictionOutcome::FP) {       //储存检索回环的帧id
            new_lc_pairs.emplace_back(ptr_cm_tgt->getIntID(), ptr_cands[0]->getIntID());
            new_lc_tfp.emplace_back(pred_res.tfpn == PredictionOutcome::TP);
          }
        }

        // 储存描述符
        ptr_contour_db->addScan(ptr_cm_tgt, info_.ts);
        ptr_contour_db->pushAndBalance(info_.seq, info_.ts); // 放入检索树

        //存放完成定位时间戳
        // pre_times.pushstamps(pt_lecd.pt_seq, 6);
        
        // 打印时间戳
        // std::cout << "index: " << pt_lecd.pt_seq << " stamps: ";
        // for(auto& stamp_ : pre_times.stamps[pt_lecd.pt_seq])
        // {
        //     std::cout << stamp_ << " ";
        // }
        // std::cout << std::endl;

        // 统计数据清空
        cur_seq++;

        std::cout << std::endl; // 完成一次识别 打印空行
    }
}

int main(int argc, char **argv)
{

    struct timeval tv;
    gettimeofday(&tv, NULL); // 获取当前时间
    long microseconds = tv.tv_sec * 1000 * 1000 + tv.tv_usec;
    std::cout << "Current time (in microseconds): " << microseconds << " long size: " << sizeof(long) << std::endl;

    std::string config_path = "/home/jtcx/remote_control/code/localization/src/lecd/include/config.yaml";
    loadConfig(config_path, fpath_outcome_sav);
    stp = SequentialTimeProfiler(fpath_outcome_sav);    //初始化顺序时间档案器
    // 创建 MQTT 接收线程
    std::thread mqtt_thread(mqtt_receiver_thread);
    // 创建定位线程
    std::thread localization_thread_instance(localization_thread);
    // 等待线程完成（实际上会一直运行）
    mqtt_thread.join();
    localization_thread_instance.join();

    return 0;
}