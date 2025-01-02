//
// Created by lewis on 5/5/22.
//

#include "contour_mng.h"

//保存椭圆数据 
void LECDManager::saveLECDs(const std::string &fpath,
                                  const std::vector<std::vector<std::shared_ptr<LECDView>>> &lecd_views) {
  // 0:level, 1:cell_cnt, 2:pos_mean, 4:pos_cov, 8:eig_vals, eig_vecs(10), 14:eccen, 15:vol3_mean, 16:com, 18,19:..
  // Note that recording data as strings has accuracy loss
//    std::string fpath = sav_dir + "/lecds_" + str_id_ + ".txt";
  std::fstream res_file(fpath, std::ios::out);

  if (res_file.rdstate() != std::ifstream::goodbit) {
    std::cerr << "Error opening " << fpath << std::endl;
    return;
  }
  printf("Writing results to file \"%s\" ...", fpath.c_str());
  res_file << "\nDATA_START\n";
  for (const auto &layer: lecd_views) {
    for (const auto &cont: layer) {
      res_file << cont->level_ << '\t';
      res_file << cont->cell_cnt_ << '\t';

      res_file << cont->pos_mean_.x() << '\t' << cont->pos_mean_.y() << '\t';
      for (int i = 0; i < 4; i++)
        res_file << cont->pos_cov_.data()[i] << '\t';

      res_file << cont->eig_vals_.x() << '\t' << cont->eig_vals_.y() << '\t';
      for (int i = 0; i < 4; i++)
        res_file << cont->eig_vecs_.data()[i] << '\t';

      res_file << cont->eccen_ << '\t';
      res_file << cont->vol3_mean_ << '\t';
      res_file << cont->com_.x() << '\t' << cont->com_.y() << '\t';

      res_file << int(cont->ecc_feat_) << '\t';
      res_file << int(cont->com_feat_) << '\t';
      res_file << int(cont->del_enable) << '\t';

      res_file << '\n';
    }
  }
  res_file << "DATA_END\n";
  res_file.close();
  printf("Writing results finished.\n");

}

void LECDManager::makeLECDs() {
//    float h_min = -VAL_ABS_INF_;
  cv::Mat last_label_img;
  int16_t lev = 0;
  for (const auto &h_min: cfg_.lv_grads_) {
    printf("Height [%f, +]\n", h_min);
    // clamp image
    if (lecd_views_.empty()) {
      cv::Mat mask, mask_u8;
      cv::threshold(bev_, mask, h_min, 255, cv::THRESH_BINARY); // mask is same type and dimension as bev_
      // 1. select points higher than a threshold
      mask.convertTo(mask_u8, CV_8U);

      cv::imwrite("cart_context-mask-" + std::to_string(lev) + "-" + str_id_ + ".png", mask_u8);

      // 2. calculate connected blobs
      cv::Mat1i labels, stats;  // int (CV_32S)
      cv::Mat centroids;
      cv::connectedComponentsWithStats(mask_u8, labels, stats, centroids, 8, CV_32S);

      // // aux: show image lecd group
      cv::Mat label_img;
      cv::normalize(labels, label_img, 0, 255, cv::NORM_MINMAX);
      cv::imwrite("cart_context-labels-" + std::to_string(lev) + "-" + str_id_ + ".png", label_img);
      cv::imwrite("cart_context-mask-" + std::to_string(lev) + "-" + str_id_ + ".png", mask_u8);

      // 3. create lecds for each connected component
      // https://stackoverflow.com/questions/37745274/opencv-find-perimeter-of-a-connected-component/48618464#48618464
      std::vector<std::shared_ptr<LECDView>> level_conts;
      for (int n = 1; n < stats.rows; n++) {  // n=0: background
        printf("Area: %d\n", stats.at<int>(n, cv::CC_STAT_AREA));

        //Rectangle around the connected component
        cv::Rect rect(stats(n, 0), stats(n, 1), stats(n, 2), stats(n, 3)); // Rect: col0, row0, n_col, n_row

//          // Get the mask for the lecd
//          cv::Mat1b mask_n = labels(rect) == n;
//          printf("countour ROI: %d, %d\n", mask_n.rows, mask_n.cols);

        RunningStatRecorder tmp_rec;
        int poi_r = -1, poi_c = -1;

        for (int i = rect.y; i < rect.y + rect.height; i++)
          for (int j = rect.x; j < rect.x + rect.width; j++)
            if (bev_(i, j) > h_min) {  // consistent with opencv threshold: if src(x,y)>thresh, ...
              tmp_rec.runningStats(i, j, bev_(i, j));
              poi_r = i;
              poi_c = j;
            }

//        std::shared_ptr<LECDView> ptr_tmp_cv(new LECDView(lev, poi_r, poi_c, nullptr));
        std::shared_ptr<LECDView> ptr_tmp_cv(new LECDView(lev, poi_r, poi_c));
        ptr_tmp_cv->calcStatVals(tmp_rec, view_stat_cfg_);
        DCHECK(ptr_tmp_cv->cell_cnt_ == stats(n, 4));
        level_conts.emplace_back(ptr_tmp_cv);
      }
      lecd_views_.emplace_back(level_conts);
    } else {
      // create children from parents (ancestral tree)
      for (auto parent: lecd_views_.back()) {

      }

    }

    lev++;
//      h_min = cap;
  }
}

void LECDManager::saveLECDImage(const std::string &fpath, int level) const {
  CHECK(!bev_.empty());
  cv::imwrite(fpath, getLECDImage(level));
}


// outdated
//std::pair<Eigen::Isometry2d, bool>
//LECDManager::calcScanCorresp(const LECDManager &src, const LECDManager &tgt) {
//  DCHECK_EQ(src.lecd_views_.size(), tgt.lecd_views_.size());
//  printf("calcScanCorresp(): \n");
//
//  // configs
//  int num_tgt_top = 5;
//  int num_src_top = 4;
//
//  int num_tgt_ser = 10; // when initial result is positive, we progressively search more correspondence pairs
//  int num_src_ser = 10;
//  std::vector<std::pair<int, int>> src_q_comb = {{0, 1},
//                                                 {0, 2},
//                                                 {0, 3},
//                                                 {1, 2},
//                                                 {1, 3},
//                                                 {2, 3}};  // in accordance with num_src_top
//
//  std::pair<Eigen::Isometry2d, bool> ret{};
//  int num_levels = (int) src.lecd_views_.size() - 2;
//
//  // TODO: check FP rate for retrieval tasks
//
//  for (int l = 0; l < num_levels; l++) {
//    if (src.lecd_views_[l].size() < num_src_top || tgt.lecd_views_[l].size() < 3)
//      continue;
//    if (ret.second)
//      break;
//    printf("Matching level: %d\n", l);
//    for (const auto &comb: src_q_comb) {
//      for (int i = 0; i < std::min((int) tgt.lecd_views_[l].size(), num_tgt_top); i++)
//        for (int j = 0; j < std::min((int) tgt.lecd_views_[l].size(), num_tgt_top); j++) {
//          if (j == i)
//            continue;
//          // LECD Correspondence Proposal: comb.first=i, comb.second=j
//          const auto sc1 = src.lecd_views_[l][comb.first], sc2 = src.lecd_views_[l][comb.second],
//              tc1 = tgt.lecd_views_[l][i], tc2 = tgt.lecd_views_[l][j];
//
////            printf("-- Check src: %d, %d, tgt: %d, %d\n", comb.first, comb.second, i, j);
//
//          // 1. test if the proposal fits in terms of individual lecds
//          bool is_pairs_sim = LECDView::checkSim(*sc1, *tc1) && LECDView::checkSim(*sc2, *tc2);
//          if (!is_pairs_sim) {
//            continue;
//          }
//
//          // 2. check geometry center distance
//          double dist_src = (sc1->pos_mean_ - sc2->pos_mean_).norm();
//          double dist_tgt = (tc1->pos_mean_ - tc2->pos_mean_).norm();
//          if (std::max(dist_tgt, dist_src) > 5.0 && diff_delt(dist_src, dist_tgt, 5.0))
//            continue;
//
//          // 3. check lecd orientation
//          V2F cent_s = (sc1->pos_mean_ - sc2->pos_mean_).normalized();
//          V2F cent_t = (tc1->pos_mean_ - tc2->pos_mean_).normalized();
//          if (sc1->ecc_feat_ && tc1->ecc_feat_) {
//            float theta_s = std::acos(cent_s.transpose() * sc1->eig_vecs_.col(1));   // acos: [0,pi)
//            float theta_t = std::acos(cent_t.transpose() * tc1->eig_vecs_.col(1));
//            if (diff_delt<float>(theta_s, theta_t, M_PI / 12) && diff_delt<float>(M_PI - theta_s, theta_t, M_PI / 12))
//              continue;
//          }
//          if (sc2->ecc_feat_ && tc2->ecc_feat_) {
//            float theta_s = std::acos(cent_s.transpose() * sc2->eig_vecs_.col(1));   // acos: [0,pi)
//            float theta_t = std::acos(cent_t.transpose() * tc2->eig_vecs_.col(1));
//            if (diff_delt<float>(theta_s, theta_t, M_PI / 6) && diff_delt<float>(M_PI - theta_s, theta_t, M_PI / 6))
//              continue;
//          }
//
//          // 4. PROSAC
//          // 4.1 get the rough transform to facilitate the similarity check (relatively large acceptance range)
//          // can come from a naive 2 point transform estimation or a gmm2gmm
//          Eigen::Matrix3d T_delta = estimateTF<double>(sc1->pos_mean_.cast<double>(), sc2->pos_mean_.cast<double>(),
//                                                       tc1->pos_mean_.cast<double>(),
//                                                       tc2->pos_mean_.cast<double>()).matrix(); // naive 2 point estimation
//
//          // for pointset transform estimation
//          Eigen::Matrix<double, 2, Eigen::Dynamic> pointset1; // src
//          Eigen::Matrix<double, 2, Eigen::Dynamic> pointset2; // tgt
//          pointset1.resize(2, 2);
//          pointset2.resize(2, 2);
//          pointset1.col(0) = sc1->pos_mean_.cast<double>();
//          pointset1.col(1) = sc2->pos_mean_.cast<double>();
//          pointset2.col(0) = tc1->pos_mean_.cast<double>();
//          pointset2.col(1) = tc2->pos_mean_.cast<double>();
//
//          // 4.2 create adjacency matrix (binary weight bipartite graph) or calculate on the go?
//          std::vector<std::pair<int, int>> match_list = {{comb.first,  i},
//                                                         {comb.second, j}};
//          std::set<int> used_src{comb.first, comb.second}, used_tgt{i, j};
//          // 4.3 check if new pairs exit
//          double tf_dist_max = 5.0;
//          for (int ii = 0; ii < std::min((int) src.lecd_views_[l].size(), num_src_ser); ii++) {
//            if (used_src.find(ii) != used_src.end())
//              continue;
//            for (int jj = 0; jj < std::min((int) tgt.lecd_views_[l].size(), num_tgt_ser); jj++) {
//              if (used_tgt.find(jj) != used_tgt.end())
//                continue;
//              V2F pos_mean_src_tf = T_delta.block<2, 2>(0, 0).cast<float>() * src.lecd_views_[l][ii]->pos_mean_
//                                    + T_delta.block<2, 1>(0, 2).cast<float>();
//              if ((pos_mean_src_tf - tgt.lecd_views_[l][jj]->pos_mean_).norm() > tf_dist_max ||
//                  !LECDView::checkSim(*src.lecd_views_[l][ii], *tgt.lecd_views_[l][jj])
//                  )
//                continue;
//              // handle candidate pairs
//              // TODO: check consensus before adding:
//              match_list.emplace_back(ii, jj);
//              used_src.insert(ii);  // greedy
//              used_tgt.insert(jj);
//
//              // TODO: update transform
//              // pure point method: umeyama
//
//              pointset1.conservativeResize(Eigen::NoChange_t(), match_list.size());
//              pointset2.conservativeResize(Eigen::NoChange_t(), match_list.size());
//              pointset1.rightCols(1) = src.lecd_views_[l][ii]->pos_mean_.cast<double>();
//              pointset2.rightCols(1) = tgt.lecd_views_[l][jj]->pos_mean_.cast<double>();
//              T_delta = Eigen::umeyama(pointset1, pointset2, false);  // also need to check consensus
//
//            }
//            // TODO: termination criteria
//          }
//
//
//          // TODO: metric results, filter out some outlier
//          if (match_list.size() > 4) {
//            printf("Found matched pairs in level %d:\n", l);
//            for (const auto &pr: match_list) {
//              printf("\tsrc:tgt  %d: %d\n", pr.first, pr.second);
//            }
//            std::cout << "Transform matrix:\n" << T_delta << std::endl;
//            // TODO: move ret to later
//            ret.second = true;
//            ret.first.setIdentity();
//            ret.first.rotate(std::atan2(T_delta(1, 0), T_delta(0, 0)));
//            ret.first.pretranslate(T_delta.block<2, 1>(0, 2));
//          }
//        }
//    }
//
//    // TODO: cross level consensus
//
//    // TODO: full metric estimation
//
//  }
//
//  return ret;
//}

//递归结构 计算各层的统计变量并存储到cont_views_ parent是上一层次的椭圆数据指针 cc_roi的初始值是bev全范围  制作cont_views_lecd_viewslecd_
void LECDManager::makeLECDRecursiveHelper(const cv::Rect &cc_roi, const cv::Mat1b &cc_mask, int level,
                                                const std::shared_ptr<LECDView> &parent) {
  DCHECK(bool(level)==bool(parent));    //判断level 与指针parent状态是否一致？ 非0/非空 || 0/空
  if (level >= cfg_.lv_grads_.size())   //判断层数 层数满足即退出
    return;

  float h_min = cfg_.lv_grads_[level], h_max = VAL_ABS_INF_;    //获取层次的上下界

  cv::Mat1f bev_roi = bev_(cc_roi), thres_roi;    //从bev中提取全范围的矩形区域赋值给bev_roi
  cv::threshold(bev_roi, thres_roi, h_min, 255, cv::THRESH_BINARY);   //大于层次下边界的被赋值为255

  cv::Mat1b bin_bev_roi;
  thres_roi.convertTo(bin_bev_roi, CV_8U);    //类型转换函数，将float转换成8位无符号，输出到bin_bev_roi

  if (level)    //在有层数时 cc_mark掩码已经被传入处理后的非0值
//      cv::bitwise_and(bin_bev_roi, bin_bev_roi, bin_bev_roi, cc_mask);  // Wrong method: some pixels may be unaltered since neglected by mask
    cv::bitwise_and(bin_bev_roi, cc_mask, bin_bev_roi);     //与cc_mask掩码进行按位与操作，并返回至bin_bev_roi 目的是什么？

  if (level < cfg_.lv_grads_.size() - 1)  //层次为非顶层时
    h_max = cfg_.lv_grads_[level - 1];    //? 这个是什么意思    //获取层次的上边界？ 如何避免level = 0 的情况

  // 2. calculate connected blobs
  cv::Mat1i labels, stats;  // int (CV_32S)    // stats 中的每一行对应一个连通组件的统计信息  labels 中的每个像素值表示该像素属于哪个连通组件（0为背景，!0为前景的连通组件）
  cv::Mat centroids;  // not in use
  //统计出来的是存在点的组件（当前层次及以上层次的组件区域和）
  cv::connectedComponentsWithStats(bin_bev_roi, labels, stats, centroids, 8, CV_32S);   // on local patch 对图像中的连通组件进行标记和统计

  // 3. create lecds for each connected component
  // https://stackoverflow.com/questions/37745274/opencv-find-perimeter-of-a-connected-component/48618464#48618464
  for (int n = 1; n < stats.rows; n++) {  // n=0: background
//      printf("Area: %d\n", stats.at<int>(n, cv::CC_STAT_AREA));
    if (stats(n, 4) < cfg_.min_lecd_cell_cnt_)  // ignore lecds that are too small   //面积太小忽略
      continue;

    //Rectangle around the connected component
    // Rect: col0, row0, n_col, n_row
    cv::Rect rect_g(stats(n, 0) + cc_roi.x, stats(n, 1) + cc_roi.y, stats(n, 2), stats(n, 3)); // global: on bev  //因为生成组件的bin_bev_roi是用矩形在bev上截下来的，rect_g则是指在全局坐标系下
    cv::Rect rect_l(stats(n, 0), stats(n, 1), stats(n, 2), stats(n, 3)); // local: relative to bev_roi    // 局部矩形

    cv::Mat1b mask_n = labels(rect_l) == n;     //提取出label中选中的矩形区域中标签为n的掩码,且这个为下一个层次的掩码,用于在下一个层次去除已经处理过的组件部分

    RunningStatRecorder tmp_rec;
    int poi_r = -1, poi_c = -1;  // the point(r,c) coordinate on the global bev for the lecd   //在全局bev下的坐标(像素)

    //遍历所有可能的椭圆？统计数据暂存至RunningStatRecorder
    for (int i = 0; i < rect_l.height; i++)
      for (int j = 0; j < rect_l.width; j++)
        if (mask_n(i, j)) {
          poi_r = i + rect_g.y;
          poi_c = j + rect_g.x;
//            tmp_rec.runningStats(i + rect_g.y, j + rect_g.x, bev_(i + rect_g.y, j + rect_g.x)); // discrete
//          V2F c_point = pillar_pos2f_.at(poi_r * cfg_.n_col_ + poi_c);

          int q_hash = poi_r * cfg_.n_col_ + poi_c;   //网格序号
          std::pair<int, Pixelf> sear_res = search_vec<Pixelf>(bev_pixfs_, 0,
                                                               (int) bev_pixfs_.size() - 1, q_hash);    //查找对应的pixelf数据
          DCHECK_EQ(sear_res.first, q_hash);
          tmp_rec.runningStatsF(sear_res.second.row_f, sear_res.second.col_f, bev_(poi_r, poi_c)); // continuous
//          tmp_rec.runningStatsF(c_point.x(), c_point.y(), bev_(poi_r, poi_c)); // continuous
        }

//    std::shared_ptr<LECDView> ptr_tmp_cv(new LECDView(level, poi_r, poi_c, parent));
    std::shared_ptr<LECDView> ptr_tmp_cv(new LECDView(level, poi_r, poi_c));    //构建新的椭圆指针
    ptr_tmp_cv->calcStatVals(tmp_rec, view_stat_cfg_);                              //计算椭圆的统计数据
    DCHECK(ptr_tmp_cv->cell_cnt_ == stats(n, 4));
    lecd_views_[level].emplace_back(ptr_tmp_cv);    // add to the manager's matrix
//    if (parent)
//      parent->children_.emplace_back(ptr_tmp_cv);

    // recurse
    // Get the mask for the lecd

//      printf("lecd ROI: %d, %d, level: %d\n", mask_n.rows, mask_n.cols, level);
    makeLECDRecursiveHelper(rect_g, mask_n, level + 1, ptr_tmp_cv);  //继续递归 这里的递归不是完全处理完一个层次的椭圆后才递归，而是在每个椭圆处理时就进行递归，由下到上

//      if (level == 2) {
//        cv::bitwise_or(mask_n, visualization(rect_g), visualization(rect_g));
//      }

  }

}
