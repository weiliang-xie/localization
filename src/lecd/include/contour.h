#ifndef CONTOUR_H
#define CONTOUR_H

#include <utility>
#include <vector>
#include <iostream>

#include <glog/logging.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <opencv2/core/types.hpp>

#include "tools/algos.h"

typedef Eigen::Matrix<float, 2, 1> V2F;
typedef Eigen::Matrix<float, 2, 2> M2F;
typedef Eigen::Matrix<double, 2, 1> V2D;
typedef Eigen::Matrix<double, 2, 2> M2D;


struct ContourViewStatConfig {
  int16_t min_cell_cov = 4;
  float point_sigma = 1.0; // have nothing to do with resolution: on pixel only
  float com_bias_thres = 0.5;  // com dist from geometric center
//  int half_strip_num_ = 4;
};

// The configuration for checking the similarity of two contours
struct ContourSimThresConfig {
  float ta_cell_cnt = 6, tp_cell_cnt = 0.2;
  float tp_eigval = 0.2;
  float ta_h_bar = 0.3;  // 0.75 for mulran
  float ta_rcom = 0.4, tp_rcom = 0.25;
  float ta_eccen = 0.08;
};

// use a separate recorder to record when creating contour view, and discard it after use.  一个椭圆内的暂存数据，用于计算椭圆的特性参数
struct RunningStatRecorder {
  int16_t cell_cnt_{};  //局部矩阵行列（这个代表了什么）总和
  V2D cell_pos_sum_;    //x y坐标的总和
  M2D cell_pos_tss_;    //坐标的模的平方的和
  float cell_vol3_{};  // or "weight" of the elevation mountain. Should we include volumns under the h_min_? 累计最大高度和 这里用高度作为质量比重 height-> weight
  V2D cell_vol3_torq_;  //乘上相应比重的 xy和

  RunningStatRecorder() {
    cell_pos_sum_.setZero();
    cell_pos_tss_.setZero();
    cell_vol3_torq_.setZero();
  }

  // TODO: call this function everytime encounters a pixel belonging to this connected component
  void runningStats(int curr_row, int curr_col, float height) {
    DCHECK_GE(curr_row, -0.5f);
    DCHECK_GE(curr_col, -0.5f);
    cell_cnt_ += 1;
    V2D v_rc(curr_row, curr_col);
    cell_pos_sum_ += v_rc;
    cell_pos_tss_ += v_rc * v_rc.transpose();
    cell_vol3_ += height;
    cell_vol3_torq_ += height * v_rc;
//    voxels_pos_.emplace_back(v_rc);  // desc cont itself (5/6)
  }
  //暂存遍历网格时的数据，用于后续的椭圆数据计算
  ///\param  curr_row image坐标系下的x坐标
  ///\param  curr_col image坐标系下的y坐标
  ///\param  curr_col 该像素的最大高度
  void runningStatsF(float curr_row, float curr_col, float height) {   // a more accurate one with continuous coordinate
    DCHECK_GE(curr_row, -0.5f);   //判断最小值？
    DCHECK_GE(curr_col, -0.5f);
    cell_cnt_ += 1;                             //计算局部矩阵行列（这个代表了什么）总和
    V2D v_rc(curr_row, curr_col);
    cell_pos_sum_ += v_rc;                      //累计xy 位姿
    cell_pos_tss_ += v_rc * v_rc.transpose();
    cell_vol3_ += height;                       //累计最大高度
    cell_vol3_torq_ += height * v_rc;
//    voxels_pos_.emplace_back(v_rc);  // desc cont itself (4/6)
  }

  static RunningStatRecorder addContourStat(const RunningStatRecorder &rec1, const RunningStatRecorder &rec2) {
    RunningStatRecorder res;
    res.cell_cnt_ = rec1.cell_cnt_ + rec2.cell_cnt_;
    res.cell_pos_sum_ = rec1.cell_pos_sum_ + rec2.cell_pos_sum_;
    res.cell_pos_tss_ = rec1.cell_pos_tss_ + rec2.cell_pos_tss_;
    res.cell_vol3_ = rec1.cell_vol3_ + rec2.cell_vol3_;
    res.cell_vol3_torq_ = rec1.cell_vol3_torq_ + rec2.cell_vol3_torq_;
    return res;
  }
};

struct ContourView {
  // Coordinate definition:
  //  row as x, col as y, center of pixel(0,0) as origin.
  //  Use (row, col) to access all the image data

  // config
//  const ContourViewStatConfig cfg_;

  // property:
  int16_t level_;   //lay level 
  int16_t poi_[2]; // a point in full bev coordinate belonging to this contour/slice. 在bev下的像素坐标？

  // statistical summary
  int16_t cell_cnt_{};    //椭圆占据的网格数量
  V2F pos_mean_;          //xy均值 中心点
  M2F pos_cov_;           //协方差矩阵  没用上
  V2F eig_vals_;          //协方差特征值
  M2F eig_vecs_; // gaussian ellipsoid axes. if ecc_feat_==false, this is meaningless  特征值对应的椭球轴(特征向量) 长轴匹配+生成协方差 ecc_feat_无效则为无意义值
  float eccen_{};   // 0: circle    //偏心率，0为圆形
  float vol3_mean_{};     //高度均值 用于统计数据计算中
  V2F com_; // center of mass 以高度为质量比重的中心
  bool ecc_feat_ = false;   // eccentricity large enough (with enough cell count)   椭圆有效（足够大，离心率足够大）判断
  bool com_feat_ = false;   // com not at fitted geometric center 质心与均值中心的差距判断，大于0.5则为1（不在同一点上）
  bool del_enable = false;

  //跑数据用
  int num_level_removed = 0;

  //xwl 统计数据大小
  const int statistical_data_size_baseline = sizeof(cell_cnt_) + sizeof(pos_mean_) + sizeof(eig_vals_) + sizeof(eig_vecs_)
                                              + sizeof(vol3_mean_) + sizeof(com_) + sizeof(ecc_feat_)
                                              + sizeof(level_) + sizeof(poi_);
  const int statistical_data_size_test = sizeof(cell_cnt_) + sizeof(pos_mean_) + sizeof(eig_vals_) + sizeof(eig_vecs_)
                                              + sizeof(eccen_) + sizeof(ecc_feat_) + sizeof(vol3_mean_)
                                              + sizeof(level_) + sizeof(poi_);

  // Raw data (the pixels that belong to this Contour. Is is necessary?)
  // TODO
  // desc cont itself (6/6)
//  std::vector<V2D> voxels_pos_;
//  std::vector<float> strip_width_;

//  // hierarchy
//  std::shared_ptr<ContourView> parent_;
//  std::vector<std::shared_ptr<ContourView>> children_;

  // TODO: 0. build a contour from 3: pic, roi, height threshold. Called in manager.
  explicit ContourView(int16_t level, int16_t poi_r, int16_t poi_c) : level_(level) {
    DCHECK_GE(poi_r, 0);
    DCHECK_GE(poi_c, 0);
    poi_[0] = poi_r;
    poi_[1] = poi_c;
  };

  ContourView(const ContourView &obj) = default;

  // TO-DO: 2. calculate statistics from running data (including feature hypothesis) 根据运行数据计算统计数据并保存（质心、均值中心、协方差、特征值与特征向量及有效性判断）
  void calcStatVals(const RunningStatRecorder &rec, const ContourViewStatConfig &cfg) {
    //计算基本统计数据
    cell_cnt_ = rec.cell_cnt_;
    pos_mean_ = rec.cell_pos_sum_.cast<float>() / cell_cnt_;

    vol3_mean_ = rec.cell_vol3_ / cell_cnt_;
    com_ = rec.cell_vol3_torq_.cast<float>() / rec.cell_vol3_;

//    strip_width_.clear();   desc cont itself (3/6)

    // eccentricity:
    if (cell_cnt_ < cfg.min_cell_cov) {     //网格数量小于限制阈值
      pos_cov_ = M2F::Identity() * cfg.point_sigma * cfg.point_sigma;   //协方差为默认值  
      eig_vals_ = V2F(cfg.point_sigma, cfg.point_sigma);                //特征值为默认
      eig_vecs_.setIdentity();
      ecc_feat_ = false;
      com_feat_ = false;
//      strip_width_.resize(cfg.half_strip_num_, 0);   desc cont itself (2/6)
    } else {
      pos_cov_ =
//          (rec.cell_pos_tss_ - rec.cell_pos_sum_ * pos_mean_.transpose() - pos_mean_ * rec.cell_pos_sum_.transpose() +
//           pos_mean_ * pos_mean_.transpose() * cell_cnt_) / (cell_cnt_ - 1);
          (rec.cell_pos_tss_.cast<float>() - pos_mean_ * pos_mean_.transpose() * cell_cnt_) /
          (cell_cnt_ - 1); // simplified, verified
      Eigen::SelfAdjointEigenSolver<M2F> es(pos_cov_.template selfadjointView<Eigen::Upper>());     //计算特征值和特征向量
      eig_vals_ = es.eigenvalues();  // increasing order
      if (eig_vals_(0) < cfg.point_sigma)  // determine if eccentricity feat using another function
        eig_vals_(0) = cfg.point_sigma;
      if (eig_vals_(1) < cfg.point_sigma)   //这里限制了特征值的最小值
        eig_vals_(1) = cfg.point_sigma;
      eccen_ = std::sqrt(eig_vals_(1) * eig_vals_(1) - eig_vals_(0) * eig_vals_(0)) / eig_vals_(1);
      eig_vecs_ = es.eigenvectors();

      ecc_feat_ = eccentricitySalient(cfg);

      // vol/weight of mountain:
      com_feat_ = centerOfMassSalient(cfg);

      // describe ellipse with ratio areas. desc cont itself (1/6)
/*      std::vector<V2D> strips;  // {perp long axis:along long axis}, since the large eig vec is the second one
      for (auto &voxels_po: voxels_pos_) {
        strips.emplace_back((voxels_po - pos_mean_).transpose() * eig_vecs_);
      }
      double strip_beg = 1e6, strip_end = -1e6;
      for (const auto &strip: strips) {
        strip_beg = strip.y() < strip_beg ? strip.y() : strip_beg;
        strip_end = strip.y() > strip_end ? strip.y() : strip_end;
      }
      strip_end += 1e-3;

      // descriptor 1: rotation invariant
//      std::vector<std::pair<double, double>> bins(cfg.half_strip_num_, {-1, -1});  // interpolation (1/2)
//      bins.front() = {0, 0};
//      bins.back() = {0, 0};

      std::vector<std::pair<double, double>> bins(cfg.half_strip_num_, {0.0, 0.0});
      std::vector<std::pair<int, int>> bins_elem_cnt(cfg.half_strip_num_, {0, 0});
      double step = (strip_end - strip_beg) / cfg.half_strip_num_;
      for (auto &strip: strips) {
        int bin_idx = std::floor((strip.y() - strip_beg) / step);
//        // case 1: use max value as feature "bit":
//        if (strip.x() >= 0)
//          bins[bin_idx].first = bins[bin_idx].first > strip.x() ? bins[bin_idx].first : strip.x();
//        else
//          bins[bin_idx].second = bins[bin_idx].second > -strip.x() ? bins[bin_idx].second : -strip.x();

        // case 2 (1/2): use mean value as feature bit
        if (strip.x() >= 0) {
          bins[bin_idx].first += strip.x();
          bins_elem_cnt[bin_idx].first++;
        } else {
          bins[bin_idx].second -= strip.x();
          bins_elem_cnt[bin_idx].second++;
        }
      }

      // case 2 (2/2)
      for (int i = 0; i < cfg.half_strip_num_; i++) {
        if (bins_elem_cnt[i].first)
          bins[i].first /= bins_elem_cnt[i].first;
        if (bins_elem_cnt[i].second)
          bins[i].second /= bins_elem_cnt[i].second;
      }

//      // // fill the -1 s, interpolation (2/2)
//      // // NOTE: we may not need interpolate, since very small ellipse are not very likely to be chosen as features.
//      int p1 = 0, p2 = 0;
//      for (int i = 1; i < cfg.half_strip_num_; i++) {    // the first and last bin always has elements (val !=-1)
//        if (bins[i].first >= 0) {
//          if (i - p1 > 1) {  // we can do w/o this if
//            double diff_lev = (bins[i].first - bins[p1].first) / (i - p1);
//            for (int j = p1 + 1; j < i; j++)
//              bins[j].first = (j - p1) * diff_lev + bins[p1].first;
//          }
//          p1 = i;
//        }
//        if (bins[i].second >= 0) {
//          if (i - p2 > 1) {
//            double diff_lev = (bins[i].second - bins[p2].second) / (i - p2);
//            for (int j = p2 + 1; j < i; j++)
//              bins[j].second = (j - p2) * diff_lev + bins[p2].second;
//          }
//          p2 = i;
//        }
//      }

      // // add
      for (int i = 0; i < cfg.half_strip_num_; i++) {
        strip_width_.emplace_back(bins[i].first + bins[cfg.half_strip_num_ - 1 - i].second);
      }
*/

    }

  }

  // TODO     // eccent 椭圆有效（足够大，离心率足够大）判断？ 条件判断 椭圆占据网格大于5 && 特征值差值大于0.2 && 最小特征值大于2.5
  inline bool eccentricitySalient(const ContourViewStatConfig &cfg) const {
    return cell_cnt_ > 5 && diff_perc<float>(eig_vals_(0), eig_vals_(1), 0.2f) && eig_vals_(1) > 2.5f;
  }

  // TODO: should have sth to do with total area  质心与均值中心的差距判断，大于0.5则为1
  inline bool centerOfMassSalient(const ContourViewStatConfig &cfg) const {
    return (com_ - pos_mean_).norm() > cfg.com_bias_thres;
  }

  // TODO
  bool orietSalient(const ContourViewStatConfig &cfg) const {
    return false;
  }


  // TODO: 3. return true if two contours can be accepted as from the same heatmap peak
  //  use normalized L2E as similarity score?
  // This is one of the checks for consensus (distributional), the other one is constellation
  // T_tgt = T_delta * T_src
//  static std::pair<Eigen::Isometry2d, bool> checkCorresp(const ContourView &cont_src, const ContourView &cont_tgt) {
  //求解统计量的相似度 返回二值量
  static bool checkSim(const ContourView &cont_src, const ContourView &cont_tgt,
                       const ContourSimThresConfig &simthres) {
//                       const ContourSimThresConfig &simthres = ContourSimThresConfig()) {
    // very loose
    // TODO: more rigorous criteria (fewer branch, faster speed)
//    std::pair<Eigen::Isometry2d, bool> ret(Eigen::Isometry2d(), false);
    bool ret = false;
    // 1. area, 2.3. eig, 4. com;
    //判断椭圆的面积差距和差异率过大
    if (diff_perc<float>(cont_src.cell_cnt_, cont_tgt.cell_cnt_, simthres.tp_cell_cnt)
        && diff_delt<float>(cont_src.cell_cnt_, cont_tgt.cell_cnt_, simthres.ta_cell_cnt)) {
#if HUMAN_READABLE
      printf("\tCell cnt not pass.\n");
#endif
      return ret;
    }

    //两个椭圆的所有特征值都大于2.0 偏心率匹配
    if(std::min(cont_src.eig_vals_(0), cont_tgt.eig_vals_(0)) > 2.0 && diff_delt<float>(cont_src.eccen_,cont_tgt.eccen_, simthres.ta_eccen)){
#if HUMAN_READABLE
      printf("\tEccen not pass.\n");
#endif
      return ret;      
    }

    //特征值1  有一个的特征值大于2（足够大）&& 特征值差异过大
    if (std::max(cont_src.eig_vals_(1), cont_tgt.eig_vals_(1)) > 2.0 &&
        diff_perc<float>(std::sqrt(cont_src.eig_vals_(1)), std::sqrt(cont_tgt.eig_vals_(1)), simthres.tp_eigval)) {
#if HUMAN_READABLE
      printf("\tBig eigval not pass.\n");
#endif
      return ret;
    }

    //特征值2  有一个的特征值大于2（足够大）&& 特征值差异过大
    if (std::max(cont_src.eig_vals_(0), cont_tgt.eig_vals_(0)) > 2.0 &&
        diff_perc<float>(std::sqrt(cont_src.eig_vals_(0)), std::sqrt(cont_tgt.eig_vals_(0)), simthres.tp_eigval)) {
#if HUMAN_READABLE
      printf("\tSmall eigval not pass.\n");
#endif
      return ret;
    }

    //网格数量大于15 && 高度均值差距过大       
    if (std::max(cont_src.cell_cnt_, cont_tgt.cell_cnt_) > 15 &&
        diff_delt<float>(cont_src.vol3_mean_, cont_tgt.vol3_mean_, simthres.ta_h_bar)) {
#if HUMAN_READABLE
      printf("\tAvg height not pass.\n");
#endif
      return ret;
    }

//     //加权质心和中心的距离的差距过大
//     const float com_r1 = (cont_src.com_ - cont_src.pos_mean_).norm();
//     const float com_r2 = (cont_tgt.com_ - cont_tgt.pos_mean_).norm();
//     if (diff_delt<float>(com_r1, com_r2, simthres.ta_rcom) && diff_perc<float>(com_r1, com_r2, simthres.tp_rcom)) {
// #if HUMAN_READABLE
//       printf("\tCom radius not pass.\n");
// #endif
//       return ret;
//     }

    ret = true;
    return ret;
  }

  // TODO: 4. add two contours. Only statistical parts are useful


  // Add contour results (**NOT** accurate statistics!)
  // procedure: revert to stat recorder and merge
  //添加轮廓结果，利用过程记录器并合并
  static ContourView
  addContourRes(const ContourView &cont1, const ContourView &cont2, const ContourViewStatConfig &cfg) {
    CHECK_EQ(cont1.level_, cont2.level_);
    RunningStatRecorder media;
    media.cell_cnt_ = cont1.cell_cnt_ + cont2.cell_cnt_;
    media.cell_pos_sum_ = (cont1.cell_cnt_ * cont1.pos_mean_ + cont2.cell_cnt_ * cont2.pos_mean_).cast<double>();
    media.cell_vol3_ = cont1.cell_cnt_ * cont1.vol3_mean_ + cont2.cell_cnt_ * cont2.vol3_mean_;
    media.cell_vol3_torq_ = (cont1.com_ * (cont1.cell_cnt_ * cont1.vol3_mean_)
                             + cont2.com_ * (cont2.cell_cnt_ * cont2.vol3_mean_)).cast<double>();
    media.cell_pos_tss_ =
        (cont1.pos_cov_ * (cont1.cell_cnt_ - 1) + cont1.cell_cnt_ * cont1.pos_mean_ * cont1.pos_mean_.transpose()
         + cont2.pos_cov_ * (cont2.cell_cnt_ - 1) + cont2.cell_cnt_ * cont2.pos_mean_ * cont2.pos_mean_.transpose()
        ).cast<double>();

    ContourView res(cont1.level_, cont1.poi_[0], cont1.poi_[1]);
    res.calcStatVals(media, cfg);

    return res;
  }

  // getter setter
//  int getArea() const {
//    return cell_cnt_;
//  }

//  void addChildren(std::shared_ptr<ContourView> &chd) {
//    children_.push_back(chd);
//  }

  // auxiliary functions
  // 1. get the position of all contour pixels
  std::vector<std::vector<int>> getContPixelPos() const {
    return {};
  }

//  // 2. visualize contour/slice
//  void displayContour(const std::string &fpath) const {
//
//  }

  //利用特征向量和特征值反向求算协方差并返回 //TODO 去除特征向量
  inline M2F getManualCov() const {
    return eig_vecs_ * eig_vals_.asDiagonal() * eig_vecs_.transpose();
    // return pos_cov_;
  }

};


#endif
