#ifndef CORRELATION_H
#define CORRELATION_H

#include "contour_mng.h"
#include <ceres/ceres.h>

#include <memory>
#include <utility>

// GMM配置
struct GMMOptConfig
{
    //  double max_corr_dist_ = 10.0; // in bev pixels
    double min_area_perc_ = 0.95;               // minimal percentage of area involved for each layer
    std::vector<int> levels_ = {1, 2, 3, 4, 5}; // the layers to be considered in ellipse gmm. 进行高斯混合的层次 不是所有的层
    double cov_dilate_scale_ = 2.0;
};

struct GMMPair
{
    struct GMMEllipse
    {
        Eigen::Matrix<double, 2, 2> cov_; // 椭圆协方差
        Eigen::Matrix<double, 2, 1> mu_;  // 椭圆中心  = 在初始化时传入的pos_mean_
        double w_;                        // contour占的网格数
        double eccen_;                    // 位姿估计使用的权重

        GMMEllipse(Eigen::Matrix<double, 2, 2> cov, Eigen::Matrix<double, 2, 1> mu, double w, double eccen) : cov_(std::move(cov)),
                                                                                                              mu_(std::move(mu)),
                                                                                                              w_(w),
                                                                                                              eccen_(eccen) {}
    };

    // layers of useful data set at the time of init
    std::vector<std::vector<GMMEllipse>> ellipses_src, ellipses_tgt;  // 各高斯混合层的 网格和占比前95%的描述符
    std::vector<std::vector<std::pair<int, int>>> selected_pair_idx_; // selected {src: tgt} pairs for f.g L2 distance calculation 经过初始位姿差值过滤的符合条件的匹配对
    // std::vector<std::vector<std::pair<int, int>>> test_selected_pair_idx_;  // selected {src: tgt} pairs for f.g L2 distance calculation 没有自适应调整的 经过初始位姿差值过滤的符合条件的匹配对
    std::vector<int> src_cell_cnts_, tgt_cell_cnts_; // 各高斯混合层的描述符网格总和
    int total_cells_src_ = 0, total_cells_tgt_ = 0;  // 所有高斯混合层的描述网格总和
    double auto_corr_src_{}, auto_corr_tgt_{};       // without normalization by cell count  所有高斯混合层内的椭圆对的自相关数
    // double ae_auto_corr_src_{}, ae_auto_corr_tgt_{};  // without normalization by cell count  没有面积作为权重的 所有高斯混合层内的椭圆对的自相关数
    // double test_auto_corr_src_{}, test_auto_corr_tgt_{};  // without normalization by cell count  没有自适应调整的 所有高斯混合层内的椭圆对的自相关数
    const double scale_;                                          // cov_dilate_scale_ 2.0 //?用于计算新的协方差矩阵，为什么要加这个权重
    std::vector<std::vector<float>> max_majax_src, max_majax_tgt; // 椭圆第二特征值
    std::vector<int> out_gmm_level;

    // 高斯混合模型初始化 储存描述符等相关数据 计算混合高斯分布的自相关性
    /// \param cm_src 候选帧描述符指针
    /// \param cm_tgt 查询帧描述符指针
    /// \param T_init should be the best possible, because we use it to simplify weak correlation pairs. 两帧之间的最优初始相对位姿值
    GMMPair(const ContourManager &cm_src, const ContourManager &cm_tgt, const GMMOptConfig &config,
            const Eigen::Isometry2d &T_init) : scale_(config.cov_dilate_scale_)
    {
        DCHECK_LE(config.levels_.size(), cm_src.getConfig().lv_grads_.size());

        // collect eigen values to isolate insignificant correlations
        // std::vector<std::vector<float>> max_majax_src, max_majax_tgt;     //椭圆第二特征值

        // 遍历高斯混合层 自调整删除高斯层
        int last_src_cnt_full = 0, last_tgt_cnt_full = 0, delete_level_cnt = 0;
        for (const auto lev : config.levels_)
        {
            int cnt_src_run = 0, cnt_src_full = cm_src.getLevTotalPix(lev);
            int cnt_tgt_run = 0, cnt_tgt_full = cm_tgt.getLevTotalPix(lev);
            // if(cnt_src_full < last_src_cnt_full / 8 || cnt_tgt_full < last_tgt_cnt_full / 8)
            // {
            //   delete_level_cnt++;
            //   // std::cout << "---------delete level " << lev << "----------"<< std::endl;
            //   out_gmm_level.emplace_back(lev);
            //   // continue;
            // }

            // last_src_cnt_full = cnt_src_full;
            // last_tgt_cnt_full = cnt_tgt_full;

            ellipses_src.emplace_back(); // 填入空元素，空的GMMEllipse容器
            ellipses_tgt.emplace_back();
            max_majax_src.emplace_back();
            max_majax_tgt.emplace_back();
            selected_pair_idx_.emplace_back();
            // test_selected_pair_idx_.emplace_back();

            const auto &src_layer = cm_src.getLevContours(lev); // 高斯分布混合不要最低的高斯分布层
            const auto &tgt_layer = cm_tgt.getLevContours(lev);

            // 存入src tgt网格占比前95%的描述符
            for (const auto &view_ptr : src_layer)
            {
                if (cnt_src_run * 1.0 / cnt_src_full >= config.min_area_perc_) // 存入的描述符网格占层内所有描述符网格的比值 >= 限制阈值 跳出for
                    break;
                ellipses_src.back().emplace_back(view_ptr->getManualCov().cast<double>(), view_ptr->pos_mean_.cast<double>(),
                                                 double(view_ptr->cell_cnt_), double(view_ptr->eccen_));
                max_majax_src.back().emplace_back(std::sqrt(view_ptr->eig_vals_.y()));
                cnt_src_run += view_ptr->cell_cnt_;
            }
            for (const auto &view_ptr : tgt_layer)
            {
                if (cnt_tgt_run * 1.0 / cnt_tgt_full >= config.min_area_perc_)
                    break;
                ellipses_tgt.back().emplace_back(view_ptr->getManualCov().cast<double>(), view_ptr->pos_mean_.cast<double>(),
                                                 double(view_ptr->cell_cnt_), double(view_ptr->eccen_));
                max_majax_tgt.back().emplace_back(std::sqrt(view_ptr->eig_vals_.y()));
                cnt_tgt_run += view_ptr->cell_cnt_;
            }
            src_cell_cnts_.emplace_back(cnt_src_full);
            tgt_cell_cnts_.emplace_back(cnt_tgt_full);
            total_cells_src_ += src_cell_cnts_.back();
            total_cells_tgt_ += tgt_cell_cnts_.back();
        }

        // pre-select (need initial guess)  利用初始位姿挑选合适的可以优化的匹配对
        int total_pairs = 0;
        double src_cnt_avg = 0;
        double src_max_three;
        double avg_gmm_ellipses_cnt = 0;
        for (int li = 0; li < ellipses_src.size(); li++)
        {
            // bool jump_level = 0;
            // for(auto out_level : out_gmm_level)
            // {
            //   // std::cout << "-----------out level: " << out_level << " current level: " << config.levels_[li] << std::endl;
            //   if(out_level == config.levels_[li])
            //   {
            //     jump_level = 1;
            //     break;
            //   }
            // }
            // if (jump_level == 1)
            // {
            //   // std::cout << "---------delete level " << config.levels_[li] << "---------- jump best selected pair " << std::endl;
            //   // continue;
            // }

            for (int si = 0; si < ellipses_src[li].size(); si++)
            {
                int index_three = 0;
                for (int ti = 0; ti < ellipses_tgt[li].size(); ti++)
                {
                    Eigen::Matrix<double, 2, 1> delta_mu = T_init * ellipses_src[li][si].mu_ - ellipses_tgt[li][ti].mu_; // 计算初始位姿下的两帧两点的偏差向量
                    //加权阈值 LECD
                    double auto_dis_w_ = 0;
                    auto_dis_w_ = 1.5 * (ellipses_tgt[li][si].w_ < ellipses_tgt[li][ti].w_ ? ellipses_tgt[li][si].w_ : ellipses_tgt[li][ti].w_) / (ellipses_tgt[li][si].w_ > ellipses_tgt[li][ti].w_ ? ellipses_tgt[li][si].w_ : ellipses_tgt[li][ti].w_);
                    auto_dis_w_ += 1.5 * (1.0 - abs(ellipses_src[li][si].eccen_ - ellipses_src[li][ti].eccen_));
                    auto_dis_w_ += 1.0;
                    if (delta_mu.norm() < auto_dis_w_ * (max_majax_src[li][si] + max_majax_tgt[li][ti]))  // close enough to correlate  距离相差两帧第二特征值的和的三倍以上则不考虑
                    // if (delta_mu.norm() < 2.0 * (max_majax_src[li][si] + max_majax_tgt[li][ti])) //CC
                    { // close enough to correlate  距离相差两帧第二特征值的和的三倍以上则不考虑
                        // if(jump_level != 1)
                        // test_selected_pair_idx_[li].emplace_back(si, ti);
                        selected_pair_idx_[li].emplace_back(si, ti);
                        total_pairs++;
                    }
                }
            }

            // 参与的平均椭圆数量
            //  avg_gmm_ellipses_cnt += double(ellipses_src[li].size()) + double(ellipses_tgt[li].size());
            //  avg_gmm_ellipses_cnt += double(ellipses_src[li].size());
        }
        // #if HUMAN_READABLE
        // double all_diff_cnt = 0;
        // double avg_diff_eccen = 0;
        // std::vector<double> diff_eccen = {};
        // std::vector<double> diff_area = {};
        // for(int i = 0; i < selected_pair_idx_.size(); i++)
        // {
        //   int diff_cnt = 0;
        //   for(int ii = 0; ii < selected_pair_idx_[i].size(); ii++)
        //   {
        //     if(ii == 0 || (ii >= 1 && selected_pair_idx_[i][ii].first != selected_pair_idx_[i][ii - 1].first) )
        //     {
        //       // std::cout << "-------TEST " << "diff cnt: " << ellipses_src[i][selected_pair_idx_[i][ii].first].w_ << std::endl;
        //       if(diff_cnt < 2)
        //         src_max_three += ellipses_src[i][selected_pair_idx_[i][ii].first].w_;
        //       src_cnt_avg += ellipses_src[i][selected_pair_idx_[i][ii].first].w_;         //平均面积
        //       diff_area.emplace_back(ellipses_src[i][selected_pair_idx_[i][ii].first].w_);
        //       avg_diff_eccen += ellipses_src[i][selected_pair_idx_[i][ii].first].eccen_;  //平均离心率
        //       diff_eccen.emplace_back(ellipses_src[i][selected_pair_idx_[i][ii].first].eccen_);
        //       diff_cnt++;
        //     }
        //   }
        //   all_diff_cnt += diff_cnt;
        // }

        // //离心率方差 标准差
        // avg_diff_eccen = avg_diff_eccen/all_diff_cnt;
        // double variance_eccen = 0;
        // for(auto diff_eccen_ : diff_eccen)
        // {
        //   variance_eccen += (diff_eccen_ - avg_diff_eccen) * (diff_eccen_ - avg_diff_eccen);
        // }
        // variance_eccen = sqrt(variance_eccen / all_diff_cnt);

        // //面积方差 标准差
        // src_cnt_avg /= all_diff_cnt;
        // double variance_area = 0;
        // for(auto diff_area_ : diff_area)
        // {
        //   variance_area += (diff_area_ - src_cnt_avg) * (diff_area_ - src_cnt_avg);
        // }
        // variance_area = sqrt(variance_area / all_diff_cnt);

        // //gmm中所有椭圆的平均数量
        // avg_gmm_ellipses_cnt /= 1.0;

        // std::cout << "-------TEST " << "src max three cnt precent: " << src_max_three/src_cnt_avg *100 << "%" << std::endl;
        // std::cout << "-------TEST " << "src avg area: " << src_cnt_avg << " src variance area: " << variance_area << std::endl;
        // std::cout << "-------TEST " << "src avg eccen: " << avg_diff_eccen << " src variance eccen: " << variance_eccen << std::endl;
        // std::cout << "-------TEST " << "all avg gmm cnt: " << avg_gmm_ellipses_cnt << " src diff ellips cnt: " << all_diff_cnt << " ";
        // tgt_cnt_avg /= double(total_pairs);

        // printf("Total pairs of gmm ellipses: %d\n", total_pairs);
        // std::cout << "-------TEST " << "src cnt avg: " << src_cnt_avg << " tgt cnt avg: " << tgt_cnt_avg << std::endl;
        // std::cout << "-------TEST " << "src cnt gap: " << src_cnt_max_gap << " tgt cnt gap: " << tgt_cnt_max_gap << std::endl;

        // #endif

        // calc auto-correlation 网格数为权重，累加各层所有椭圆对的概率密度乘积，用于计算所有高斯混合层的自相关性(相关度的分母部分)
        // for (int li = 0; li < config.levels_.size() - delete_level_cnt; li++) {
        for (int li = 0; li < config.levels_.size(); li++)
        {
            // bool jump_level = 0;
            // for(auto out_level : out_gmm_level)
            // {
            //   // std::cout << "-----------out level: " << out_level << " current level: " << config.levels_[li] << std::endl;
            //   if(out_level == config.levels_[li])
            //   {
            //     jump_level = 1;
            //     break;
            //   }
            // }

            // if (jump_level == 1)
            // {
            //   // std::cout << "---------delete level " << config.levels_[li] << "---------- jump auto corr " << std::endl;
            //   // continue;
            // }
            for (int i = 0; i < ellipses_src[li].size(); i++)
            {
                for (int j = 0; j < ellipses_src[li].size(); j++)
                {
                    Eigen::Matrix2d new_cov = scale_ * (ellipses_src[li][i].cov_ + ellipses_src[li][j].cov_); //?为什么协方差需要相加并加权重？
                    Eigen::Vector2d new_mu = ellipses_src[li][i].mu_ - ellipses_src[li][j].mu_;
                    // if(jump_level != 1)
                    // test_auto_corr_src_ += ellipses_src[li][i].w_ * ellipses_src[li][j].w_ / std::sqrt(new_cov.determinant()) *
                    //                 std::exp(-0.5 * new_mu.transpose() * new_cov.inverse() * new_mu);   //计算各个中心点在高斯混合模型下的概率
                    auto_corr_src_ += ellipses_src[li][i].w_ * ellipses_src[li][j].w_ / std::sqrt(new_cov.determinant()) *
                                      std::exp(-0.5 * new_mu.transpose() * new_cov.inverse() * new_mu); // 计算各个中心点在高斯混合模型下的概率
                    // no_cell_auto_corr_src_ += 1.0 / std::sqrt(new_cov.determinant()) *
                    //                   std::exp(-0.5 * new_mu.transpose() * new_cov.inverse() * new_mu);   //计算各个中心点在高斯混合模型下的概率 无权重

                    // double cell_cnt_w_ = (ellipses_src[li][i].w_ < ellipses_src[li][j].w_ ? ellipses_src[li][i].w_ : ellipses_src[li][j].w_) / (ellipses_src[li][i].w_ > ellipses_src[li][j].w_ ? ellipses_src[li][i].w_ : ellipses_src[li][j].w_);
                    // double eccen_w_ = 1.0 - abs(ellipses_src[li][i].eccen_ - ellipses_src[li][j].eccen_);
                    // // std::cout << "-------TEST  auto corr src" << "cell cnt weight: " << cell_cnt_w_ << " eccen weight: " << eccen_w_ << std::endl;
                    // ae_auto_corr_src_ += cell_cnt_w_ * eccen_w_ * 1.0 / std::sqrt(new_cov.determinant()) *
                    //                   std::exp(-0.5 * new_mu.transpose() * new_cov.inverse() * new_mu);   //计算各个中心点在高斯混合模型下的概率 面积和离心率差作为权重
                }
            }
            for (int i = 0; i < ellipses_tgt[li].size(); i++)
            {
                for (int j = 0; j < ellipses_tgt[li].size(); j++)
                {
                    Eigen::Matrix2d new_cov = scale_ * (ellipses_tgt[li][i].cov_ + ellipses_tgt[li][j].cov_);
                    Eigen::Vector2d new_mu = ellipses_tgt[li][i].mu_ - ellipses_tgt[li][j].mu_;
                    // if(jump_level != 1)
                    // test_auto_corr_tgt_ += ellipses_tgt[li][i].w_ * ellipses_tgt[li][j].w_ / std::sqrt(new_cov.determinant()) *
                    //                 std::exp(-0.5 * new_mu.transpose() * new_cov.inverse() * new_mu);
                    auto_corr_tgt_ += ellipses_tgt[li][i].w_ * ellipses_tgt[li][j].w_ / std::sqrt(new_cov.determinant()) *
                                      std::exp(-0.5 * new_mu.transpose() * new_cov.inverse() * new_mu);
                    // no_cell_auto_corr_tgt_ += 1.0 / std::sqrt(new_cov.determinant()) *
                    //                   std::exp(-0.5 * new_mu.transpose() * new_cov.inverse() * new_mu);

                    // double cell_cnt_w_ = (ellipses_tgt[li][i].w_ < ellipses_tgt[li][j].w_ ? ellipses_tgt[li][i].w_ : ellipses_tgt[li][j].w_) / (ellipses_tgt[li][i].w_ > ellipses_tgt[li][j].w_ ? ellipses_tgt[li][i].w_ : ellipses_tgt[li][j].w_);
                    // double eccen_w_ = 1.0 - abs(ellipses_tgt[li][i].eccen_ - ellipses_tgt[li][j].eccen_);
                    // ae_auto_corr_tgt_ += cell_cnt_w_ * eccen_w_ * 1.0 / std::sqrt(new_cov.determinant()) *
                    //                   std::exp(-0.5 * new_mu.transpose() * new_cov.inverse() * new_mu);
                }
            }
        }
        //    printf("Auto corr: src: %f, tgt: %f\n", auto_corr_src_, auto_corr_tgt_);
        //    printf("Tot cells: src: %d, tgt: %d\n", total_cells_src_, total_cells_tgt_);
    }

    // evaluate
    // 残差函数 优化出来的是以src为基底在image坐标系下的转移矩阵
    template <typename T>
    bool operator()(const T *parameters, T *cost) const
    {

        const T x = parameters[0];
        const T y = parameters[1];
        const T theta = parameters[2];

        T add_opimite = T(0);
        int add_num = 0;

        Eigen::Matrix<T, 2, 2> R;
        R << cos(theta), -sin(theta), sin(theta), cos(theta);
        Eigen::Matrix<T, 2, 1> t(x, y);

        cost[0] = T(0);

        // 遍历各层的匹配对
        for (int li = 0; li < selected_pair_idx_.size(); li++)
        {
            for (const auto &pr : selected_pair_idx_[li])
            {
                // TODO: fine tuning: different weights for different levels
                Eigen::Matrix<T, 2, 2> new_cov =
                    scale_ * (R * ellipses_src[li][pr.first].cov_ * R.transpose() + ellipses_tgt[li][pr.second].cov_);    // 转换后的协方差加上tgt的协方差
                Eigen::Matrix<T, 2, 1> new_mu = R * ellipses_src[li][pr.first].mu_ + t - ellipses_tgt[li][pr.second].mu_; // 转换后的中心加上tgt的中心

                T qua = -0.5 * new_mu.transpose() * new_cov.inverse() * new_mu;
                // cost[0] += -ellipses_tgt[li][pr.second].w_ * ellipses_src[li][pr.first].w_ * 1.0 / sqrt(new_cov.determinant()) *
                //            exp(qua); // 用高斯分布公式计算残差 CC
                // add_opimite += new_mu.norm() / T(3.0 * (max_majax_src[li][pr.first] + max_majax_tgt[li][pr.second]));
                // cost[0] += -ellipses_tgt[li][pr.second].w_ * ellipses_src[li][pr.first].w_ * 0.45 * 1.0 / sqrt(new_cov.determinant()) *
                //            exp(qua) + 0.55 * add_opimite;      //残差使用GMM的L2距离的第二项，用高斯分布公式计算残差 + 中心L2
                // cost[0] += ellipses_tgt[li][pr.second].w_ * ellipses_src[li][pr.first].w_ * 1.0 / sqrt(new_cov.determinant()) *
                //            exp(qua);      //残差使用GMM的L2距离的第二项，用高斯分布公式计算残差
                // cost[0] += 1.0 / sqrt(new_cov.determinant()) * exp(qua);      //残差使用GMM的L2距离的第二项，用高斯分布公式计算残差 去除原权重

                // 面积和离心率作为权重 LECD
                 double cell_cnt_w_ = (ellipses_src[li][pr.first].w_ < ellipses_tgt[li][pr.second].w_ ? ellipses_src[li][pr.first].w_ : ellipses_tgt[li][pr.second].w_) / (ellipses_src[li][pr.first].w_ > ellipses_tgt[li][pr.second].w_ ? ellipses_src[li][pr.first].w_ : ellipses_tgt[li][pr.second].w_);
                 double eccen_w_ = 1.0 - abs(ellipses_src[li][pr.first].eccen_ - ellipses_tgt[li][pr.second].eccen_);
                 cost[0] += -cell_cnt_w_ * eccen_w_ * 1.0 / sqrt(new_cov.determinant()) * exp(qua);      //残差使用GMM的L2距离的第二项，用高斯分布公式计算残差
                //  cost[0] += -eccen_w_ * 1.0 / sqrt(new_cov.determinant()) * exp(qua);      //残差使用GMM的L2距离的第二项，用高斯分布公式计算残差
                //  cost[0] += -1.0 / sqrt(new_cov.determinant()) * exp(qua);      //残差使用GMM的L2距离的第二项，用高斯分布公式计算残差
                //  cost[0] += -ellipses_tgt[li][pr.second].w_ * ellipses_src[li][pr.first].w_ * 1.0 / sqrt(new_cov.determinant()) * exp(qua);      //用高斯分布公式计算残差
            }
            // add_num += selected_pair_idx_[li].size();
        }

        // add_opimite /= T(add_num);
        // std::cout << "========= distribute L2: " << std::sqrt(auto_corr_src_ * auto_corr_tgt_) - cost[0]
        //           << " miu L2: " << add_opimite * std::sqrt(auto_corr_src_ * auto_corr_tgt_) << std::endl;
        // cost[0] = 0.6 * (std::sqrt(auto_corr_src_ * auto_corr_tgt_) - cost[0]) + 0.4 * add_opimite * std::sqrt(auto_corr_src_ * auto_corr_tgt_);
        // cost[0] = 0.5 * (std::sqrt(ae_auto_corr_src_ * ae_auto_corr_tgt_) - cost[0]) + 0.5 * add_opimite * std::sqrt(ae_auto_corr_src_ * ae_auto_corr_tgt_);
        // cost[0] = 0.7 * (std::sqrt(ae_auto_corr_src_ * ae_auto_corr_tgt_) - cost[0]) + 0.3 * add_opimite * std::sqrt(ae_auto_corr_src_ * ae_auto_corr_tgt_);
        // cost[0] = (std::sqrt(ae_auto_corr_src_ * ae_auto_corr_tgt_) - cost[0]);
        // 验证 μ与权重
        // cost[0] = add_opimite;

        // cost[0] = 0.5 * (cost[0] / std::sqrt(auto_corr_src_ * auto_corr_tgt_))+ 0.5 * add_opimite;
        // cost[0] = 0.45 * cost[0] + 0.55 * 0.001 * add_opimite;

        return true;
    }
};

//! Constellation correlation 结构群相关性类
class ConstellCorrelation
{
    GMMOptConfig cfg_; // 初始化参数
    std::unique_ptr<ceres::GradientProblem> problem_ptr = nullptr;
    GMMPair *gmm_ptr_ = nullptr;
    double auto_corr_src{}, auto_corr_tgt{};
    // double test_auto_corr_src{}, test_auto_corr_tgt{};
    Eigen::Isometry2d T_best_; // 初始化设置成单位矩阵 位姿优化前被赋值为初始相对位姿 完成位姿优化后更新

public:
    ConstellCorrelation() = default;

    explicit ConstellCorrelation(GMMOptConfig cfg) : cfg_(std::move(cfg))
    {
        T_best_.setIdentity();
    };

    /// Split init cost calc from full optimization, in case of too many candidates  初始化GMMPair 计算初始位姿T_init的相关数
    /// \param cm_src 候选帧描述符指针
    /// \param cm_tgt 查询帧描述符指针
    /// \param T_init should be the best possible, because we use it to simplify weak correlation pairs. 两帧之间的最优初始相对位姿值
    /// \return -(T_init残差 除以 src和tgt的自相关数的乘积的开方)
    double initProblem(const ContourManager &cm_src, const ContourManager &cm_tgt, const Eigen::Isometry2d &T_init)
    {
        //    printf("Param before opt:\n");
        //    for (auto dat: parameters) {
        //      std::cout << dat << std::endl;
        //    }
        //    std::cout << T_delta.matrix() << std::endl;

        T_best_ = T_init;                                                                      // 将image坐标系下的转移矩阵作为初值
        std::unique_ptr<GMMPair> ptr_gmm_pair(new GMMPair(cm_src, cm_tgt, cfg_, T_init));      // 初始化
        std::unique_ptr<GMMPair> ptr_gmm_pair_copy(new GMMPair(cm_src, cm_tgt, cfg_, T_init)); // 初始化
        gmm_ptr_ = ptr_gmm_pair_copy.release();
        auto_corr_src = ptr_gmm_pair->auto_corr_src_;
        auto_corr_tgt = ptr_gmm_pair->auto_corr_tgt_;
        // test_auto_corr_src = ptr_gmm_pair->test_auto_corr_src_;
        // test_auto_corr_tgt = ptr_gmm_pair->test_auto_corr_tgt_;
        problem_ptr = std::make_unique<ceres::GradientProblem>(
            new ceres::AutoDiffFirstOrderFunction<GMMPair, 3>(ptr_gmm_pair.release())); // 利用GMMPair创建位姿非线性优化指针 release()释放ptr_gmm_pair的堆内存，并交给AutoDiffFirstOrderFunction

        return tryProblem(T_init);
    }

    /// Get the correlation under a certain transform  获取某个位姿下的相关性系数（残差与自相关数的计算）
    /// \param T_try The TF under which to get the correlation. Big diff between T_init and T_try will cause problems.
    /// \return -(残差 除以 src和tgt的自相关数的乘积的开方)
    double tryProblem(const Eigen::Isometry2d &T_try) const
    {
        DCHECK(problem_ptr);
        double parameters[3] = {T_try(0, 2), T_try(1, 2), std::atan2(T_try(1, 0), T_try(0, 0))}; //
        double cost[1] = {0};
        problem_ptr->Evaluate(parameters, cost, nullptr); // 计算在当前位姿的残差值
        double corr = countCorrelation(gmm_ptr_, parameters);
        // std::cout << "init try corr: " << corr << std::endl;
        return corr; // 标准化残差 返回-(残差 除以 src和tgt的自相关数的乘积的开方)
    }

    double countCorrelation(GMMPair *gmm_ptr, double *parameters) const
    {
        const double x = parameters[0];
        const double y = parameters[1];
        const double theta = parameters[2];

        Eigen::Matrix<double, 2, 2> R;
        R << cos(theta), -sin(theta), sin(theta), cos(theta);
        Eigen::Matrix<double, 2, 1> t(x, y);

        double best_match = 0;

        for (int li = 0; li < gmm_ptr->selected_pair_idx_.size(); li++)
        {
            for (const auto &pr : gmm_ptr->selected_pair_idx_[li])
            {
                // TODO: fine tuning: different weights for different levels
                Eigen::Matrix<double, 2, 2> new_cov =
                    gmm_ptr->scale_ * (R * gmm_ptr->ellipses_src[li][pr.first].cov_ * R.transpose() + gmm_ptr->ellipses_tgt[li][pr.second].cov_); // 转换后的协方差加上tgt的协方差
                Eigen::Matrix<double, 2, 1> new_mu = R * gmm_ptr->ellipses_src[li][pr.first].mu_ + t - gmm_ptr->ellipses_tgt[li][pr.second].mu_;  // 转换后的中心加上tgt的中心

                double qua = -0.5 * new_mu.transpose() * new_cov.inverse() * new_mu;
                best_match += gmm_ptr->ellipses_tgt[li][pr.second].w_ * gmm_ptr->ellipses_src[li][pr.first].w_ * 1.0 / sqrt(new_cov.determinant()) *
                              exp(qua); // 用高斯分布公式计算残差
            }
        }

        return best_match / std::sqrt(auto_corr_src * auto_corr_tgt);
        // return best_match / std::sqrt(test_auto_corr_src * test_auto_corr_tgt);
    }

    // T_tgt (should)= T_delta * T_src
    // 位姿优化函数  返回 相关度(残差 除以 src和tgt的自相关数的乘积的开方) - 优化位姿 这里的相关度只更新残差值，其他变量不变
    std::pair<double, Eigen::Isometry2d> calcCorrelation()
    {
        DCHECK(nullptr != problem_ptr);
        // gmmreg, rigid, 2D.
        double parameters[3] = {T_best_(0, 2), T_best_(1, 2),
                                std::atan2(T_best_(1, 0),
                                           T_best_(0, 0))}; // set according to the constellation output. 来自于anchor群的平均值

        ceres::GradientProblemSolver::Options options;
        options.minimizer_progress_to_stdout = HUMAN_READABLE;
        // options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 10;
        ceres::GradientProblemSolver::Summary summary;
        ceres::Solve(options, *problem_ptr, parameters, &summary); // 这里优化位姿估计
        // std::cout << summary.FullReport() << std::endl;
        // //打印优化中调用的次数
        // static float all_times = 0;
        // static float all_seconds = 0;
        // static float cere_times = 0;
        // cere_times++;
        // all_times += summary.num_gradient_evaluations;
        // all_seconds += summary.total_time_in_seconds;
        // std::cout << "\t" << "times: " << summary.num_gradient_evaluations << " seconds: " << summary.total_time_in_seconds << std::endl;
        // std::cout << "\t" << "\t" << "all times: " << all_times << " all seconds: " << all_seconds << " all cere times: " << cere_times << std::endl;

#if HUMAN_READABLE
        std::cout << summary.FullReport() << "\n";

        printf("Param after opt:\n");
        for (auto dat : parameters)
        {
            std::cout << dat << std::endl;
        }
#endif

        // normalize the score according to cell counts, and return the optimized parameter
        //    Eigen::Isometry2d T_res;
        T_best_.setIdentity();
        T_best_.rotate(parameters[2]);
        T_best_.pretranslate(V2D(parameters[0], parameters[1]));

        double corr = countCorrelation(gmm_ptr_, parameters);
        // std::cout << "last corr: " << corr << " " << parameters[0] << " " << parameters[1] << " " << parameters[2] << std::endl;
        //    printf("Correlation: %f\n", correlation);
        return {corr, T_best_};
    }

    // TODO: evaluate metric estimation performance given the 3D gt poses
    /// \param T_delta    估计位姿
    /// \param gt_src_3d  候选帧pose真值
    /// \param gt_tgt_3d  查询帧pose真值
    /// \param bev_config 阈值结构体
    /// \return 估计值与真值的误差
    static Eigen::Isometry2d evalMetricEst(const Eigen::Isometry2d &T_delta, const Eigen::Isometry3d &gt_src_3d,
                                           const Eigen::Isometry3d &gt_tgt_3d, const ContourManagerConfig &bev_config)
    {
        // ignore non-square resolution for now:
        CHECK_EQ(bev_config.reso_row_, bev_config.reso_col_);

        // 转换坐标系
        Eigen::Isometry2d T_so_ssen = Eigen::Isometry2d::Identity(), T_to_tsen; // {}_sensor in {}_bev_origin frame
        T_so_ssen.translate(V2D(bev_config.n_row_ / 2 - 0.5, bev_config.n_col_ / 2 - 0.5));
        T_to_tsen = T_so_ssen;
        Eigen::Isometry2d T_tsen_ssen2_est = T_to_tsen.inverse() * T_delta * T_so_ssen; // 将转换矩阵转换到激光雷达坐标系下
        T_tsen_ssen2_est.translation() *= bev_config.reso_row_;
        //    std::cout << "Estimated src in tgt sensor frame:\n" << T_tsen_ssen2_est.matrix() << std::endl;

        // Lidar sensor src in the lidar tgt frame, T_wc like.
        Eigen::Isometry3d T_tsen_ssen3 = gt_tgt_3d.inverse() * gt_src_3d; //?求真值坐标系之间的变换矩阵，按道理这个变换矩阵应该是src*tgt.inverse()
                                                                          //    std::cout << "gt src in tgt sensor frame 3d:\n" << T_tsen_ssen3.matrix() << std::endl;

        // TODO: project gt 3d into some gt 2d, and use
        Eigen::Isometry2d T_tsen_ssen2_gt;
        T_tsen_ssen2_gt.setIdentity();
        // for translation: just the xy difference
        // for rotation: rotate so that the two z axis align
        Eigen::Vector3d z0(0, 0, 1);
        Eigen::Vector3d z1 = T_tsen_ssen3.matrix().block<3, 1>(0, 2);
        Eigen::Vector3d ax = z0.cross(z1).normalized();
        double ang = (1 - z0.dot(z1) < 0.0000001) ? 0 : (acos(z0.dot(z1))); // 求解三维旋转矩阵z列与z0之间的夹角，生成后面的旋转向量模块矩阵
        Eigen::AngleAxisd d_rot(-ang, ax);                                  // 旋转向量模块 初始化 -ang是旋转角，ax是旋转轴 用于将三维矩阵转换成二维矩阵

        Eigen::Matrix3d R_rectified = d_rot.matrix() * T_tsen_ssen3.matrix().topLeftCorner<3, 3>(); // only top 2x2 useful 去除垂直方向上的旋转
        // std::cout << "R_rect:\n" << R_rectified << std::endl;
        CHECK_LT(R_rectified.row(2).norm(), 1 + 1e-3);
        CHECK_LT(R_rectified.col(2).norm(), 1 + 1e-3);

        T_tsen_ssen2_gt.rotate(std::atan2(R_rectified(1, 0), R_rectified(0, 0)));
        T_tsen_ssen2_gt.pretranslate(Eigen::Vector2d(T_tsen_ssen3.translation().segment(0, 2))); // only xy

        // std::cout << "T delta gt 2d:\n"
        //           << T_tsen_ssen2_gt.matrix() << std::endl; // Note T_delta is not comparable to this

        Eigen::Isometry2d T_gt_est = T_tsen_ssen2_gt.inverse() * T_tsen_ssen2_est;
        return T_gt_est; // 这是gt值与estimate值之间的旋转平移矩阵，越小越接近
    }

    /// Get estimated transform between sensors.  获取lidar坐标系下的转换矩阵 image转换矩阵 -> lidar转换矩阵
    ///  T_tgt = T_delta * T_src, image orig frame, while this one is in sensor frame
    /// \param T_delta  在image坐标系下的src到tgt的转换矩阵
    /// \param bev_config 配置参数
    /// \return idar坐标系下的转换矩阵
    static Eigen::Isometry2d getEstSensTF(const Eigen::Isometry2d &T_delta, const ContourManagerConfig &bev_config)
    {
        // ignore non-square resolution for now:
        CHECK_EQ(bev_config.reso_row_, bev_config.reso_col_);

        Eigen::Isometry2d T_so_ssen = Eigen::Isometry2d::Identity(), T_to_tsen;             // {}_sensor in {}_bev_origin frame 这两个分别是src和tgt帧的lidar-> image的转换矩阵
        T_so_ssen.translate(V2D(bev_config.n_row_ / 2 - 0.5, bev_config.n_col_ / 2 - 0.5)); // lidar 坐标系转换成 image 坐标系
        T_to_tsen = T_so_ssen;
        // 右乘T_so_ssen：将需要变换的src点先转换成image坐标系下，T_to_tsen.inverse()：将完成变换后的点从image坐标系转换到lidar坐标系
        Eigen::Isometry2d T_tsen_ssen2_est = T_to_tsen.inverse() * T_delta * T_so_ssen; // sensor in sensor frame: dist
        return T_tsen_ssen2_est;
    }
};

//! Full bev correlation
class BEVCorrelation
{
};

#endif  // CORRELATION_H
