//
// Created by lewis on 5/6/22.
//

#include "contour_db.h"

// outdated
//void ContourDB::queryKNN(const ContourManager &q_cont, std::vector<std::shared_ptr<const ContourManager>> &cand_ptrs,
//                         std::vector<KeyFloatType> &dist_sq) const {
//  cand_ptrs.clear();
//  dist_sq.clear();
//
//  // for each layer
//  std::vector<std::pair<IndexOfKey, KeyFloatType>> res_container;
//  for (int ll = 0; ll < q_levels_.size(); ll++) {
//    size_t size_beg = res_container.size();
//    for (const auto &permu_key: q_cont.getLevRetrievalKey(q_levels_[ll])) {
//      if (permu_key.sum() != 0) {
//        std::vector<std::pair<IndexOfKey, KeyFloatType>> tmp_res;
//
//        layer_db_[ll].layerKNNSearch(permu_key, cfg_.max_candi_per_layer_, cfg_.max_dist_sq_, tmp_res);
//        res_container.insert(res_container.end(), tmp_res.begin(),
//                             tmp_res.end());  // Q:different thres for different levels?
//      }
//    }
//    // limit number of returned values in layer
//    std::sort(res_container.begin() + size_beg, res_container.end(),
//              [&](const std::pair<IndexOfKey, KeyFloatType> &a, const std::pair<IndexOfKey, KeyFloatType> &b) {
//                return a.second < b.second;
//              });
//    if (res_container.size() > cfg_.max_candi_per_layer_ + size_beg)
//      res_container.resize(cfg_.max_candi_per_layer_ + size_beg, res_container[0]);
//
//  }
//
////    // limit number of returned values as whole
////    std::sort(res_container.begin(), res_container.end(),
////              [&](const std::pair<size_t, KeyFloatType> &a, const std::pair<size_t, KeyFloatType> &b) {
////                return a.second < b.second;
////              });
////    if (res_container.size() > cfg_.max_total_candi_)
////      res_container.resize(cfg_.max_total_candi_);
//
//  std::set<size_t> used_gidx;
//  printf("Query dists_sq: ");
//  for (const auto &res: res_container) {
//    if (used_gidx.find(res.first.gidx) == used_gidx.end()) {
//      used_gidx.insert(res.first.gidx);
//      cand_ptrs.emplace_back(all_bevs_[res.first.gidx]);
//      dist_sq.emplace_back(res.second);
//      printf("%7.4f", res.second);
//    }
//  }
//  printf("\n");
//
//
//  // find which bucket ranges does the query fall in
//
//  // query and accumulate results
//
//}

//重建树 建立在层级上？
/// \param idx_t1  处理后为0~4 与点云帧序号有关
/// \param curr_ts 当前时间戳
void LayerDB::rebuild(int idx_t1, double curr_ts) {
//    int idx_t1 = std::abs(seed) % (2 * (max_num_backets_ - 2));
//    if (idx_t1 > (max_num_backets_ - 2))
//      idx_t1 = 2 * (max_num_backets_ - 2) - idx_t1;

  DCHECK_LT(idx_t1, buckets_.size() - 1);
  TreeBucket &tr1 = buckets_[idx_t1], &tr2 = buckets_[idx_t1 + 1];

  DCHECK_EQ(tr1.buc_end_, tr2.buc_beg_);    //当前bucket的end与下一个bucket的beg是否相等
  bool pb1 = tr1.needPopBuffer(curr_ts), pb2 = tr2.needPopBuffer(curr_ts); // if we need to pop buffer
  if (!pb1 && !pb2)
    return;  // rebuild is scheduled at the time when we pop the buffer  没有需要pop的buffer 直接返回

  int sz1 = tr1.getTreeSize(), sz2 = tr2.getTreeSize();
  double diff_ratio = 1.0 * std::abs(sz1 - sz2) / std::max(sz1, sz2);   //求数量的差异比值
  if (pb1 && !pb2 && (diff_ratio < imba_diff_ratio_ || std::max(sz1, sz2) < min_elem_split_)) {   //有重建需求并满足数量和差异要求
    tr1.popBufferMax(curr_ts);    //压入并重建
    return;
  }

  if (!pb1 && pb2 && (diff_ratio < imba_diff_ratio_ || std::max(sz1, sz2) < min_elem_split_)) {
    tr2.popBufferMax(curr_ts);    //压入并重建
    return;
  }
  // else, balance two trees
  // 1. find new boundary/pivot.
  // q: How to find the elements to move in O(n)
  //  a1: flatten and rebuild?
  //  a2: sort the larger one, and copy parts to the smaller one.

  if (diff_ratio < 0.5 * imba_diff_ratio_) {    //差异足够小，直接传入数据
    if (pb1) tr1.popBufferMax(curr_ts);
    if (pb2) tr2.popBufferMax(curr_ts);
    return;
  }

  printf(" (m->)"); // merging here
  //TODO 这段可以优化
  if (sz1 > sz2) {
//      DCHECK_GE(sz1, min_elem_split_);
    //选择三个阈值的依据是保证在这个区间内任意分割都能达到上面的差异阈值要求 这三个阈值组成的区间是可以分割的区间，从中间出发往两边寻找合适的分割点，使得两个bucket的差异满足要求
    int to_move_max = int((sz1 - sz2 + imba_diff_ratio_ * sz2) / (2 - imba_diff_ratio_));
    int to_move_mid = int((sz1 - sz2) / 2.0);
    int to_move_min = std::max(0, int((sz1 - sz2 - imba_diff_ratio_ * sz1) / (2 - imba_diff_ratio_)));
    DCHECK_GE(to_move_max, to_move_mid);
    DCHECK_GE(to_move_mid, to_move_min);

    std::vector<int> sort_permu(sz1);
    std::iota(sort_permu.begin(), sort_permu.end(), 0);
    std::sort(sort_permu.begin(), sort_permu.end(), [&](const int &a, const int &b) {
      return tr1.data_tree_[a](bucket_chann_) < tr1.data_tree_[b](bucket_chann_); //根据key(0)的大小返回对应序号，小的序号在前
    }); // not actually moving data positions

    int num_to_move = 0;
    KeyFloatType split_val = tr1.buc_end_; // tree1: ( , split val], tree2: [split val, )  分割key值
    if (tr1.data_tree_[sort_permu[sz1 - to_move_mid]](bucket_chann_) !=
        tr1.data_tree_[sort_permu[sz1 - to_move_mid - 1]](bucket_chann_)) { // if no contagious value across split  回退to_move_mid个的key值是单独的 可以分割
      num_to_move = to_move_mid;
      split_val = tr1.data_tree_[sort_permu[sz1 - to_move_mid]](bucket_chann_);
    } else {
      KeyFloatType contagious_val = tr1.data_tree_[sort_permu[sz1 - to_move_mid]](bucket_chann_);
      int i = to_move_mid - 1;
      for (; i > to_move_min; i--) {  // look into larger side
        if (tr1.data_tree_[sort_permu[sz1 - i]](bucket_chann_) != contagious_val) {  // found the split position 往回寻找单独的key值 直到找到to_move_min
          num_to_move = i;
          split_val = tr1.data_tree_[sort_permu[sz1 - i]](bucket_chann_);
          break;
        }
      }
      if (num_to_move == 0) {   //往另一个方向寻找
        i = to_move_mid + 1;
        for (; i < to_move_max; i++) {  // look into smaller side
          if (tr1.data_tree_[sort_permu[sz1 - i]](bucket_chann_) != contagious_val) {  // found the split position
            num_to_move = i - 1;
            split_val = contagious_val;
            break;
          }
        }
      }
    }

    if (num_to_move == 0) { // should rebalance but cannot, due to a strip of contagious bucket values 有连续一致的key值，无法重建
      printf("Cannot split due to contagious values A");
      tr1.popBufferMax(curr_ts);    //?直接全部pop进，等下一次重建再分割
      if (pb2) tr2.popBufferMax(curr_ts); //若有，也pop进
      return;
    }

    for (int i = 0; i < num_to_move; i++) {  // add to shorter tree  调整 balance
      tr2.data_tree_.emplace_back(tr1.data_tree_[sort_permu[sz1 - i - 1]]);
      tr2.gkidx_tree_.emplace_back(tr1.gkidx_tree_[sort_permu[sz1 - i - 1]]);
    }

    int p_dat = sz1 - 1, p_perm = sz1 - 1;
    for (; p_perm >= sz1 - num_to_move; p_perm--) {  // rotate to the back; 排序并删除
      while (tr1.data_tree_[p_dat](bucket_chann_) >= split_val)
        p_dat--;
      if (sort_permu[p_perm] < p_dat) {
        std::swap(tr1.data_tree_[p_dat], tr1.data_tree_[sort_permu[p_perm]]);
        std::swap(tr1.gkidx_tree_[p_dat], tr1.gkidx_tree_[sort_permu[p_perm]]);
        p_dat--;
      }
    }
    DCHECK_EQ(p_dat + num_to_move, sz1 - 1);
    tr1.data_tree_.resize(p_dat + 1);
    tr1.gkidx_tree_.resize(p_dat + 1, tr1.gkidx_tree_[0]);

    // move buffer: merge and redistribute
    int p1 = 0, p2 = tr1.buffer_.size() - 1, sz_rem;    //排序buffer
    while (p1 <= p2) {
      if (tr1.buffer_[p1].pt(bucket_chann_) >= split_val && tr1.buffer_[p2].pt(bucket_chann_) < split_val) {
        std::swap(tr1.buffer_[p1], tr1.buffer_[p2]);
        p1++;
        p2--;
      } else {
        if (tr1.buffer_[p2].pt(bucket_chann_) >= split_val)
          p2--;
        if (tr1.buffer_[p1].pt(bucket_chann_) < split_val)
          p1++;
      }
    }
    sz_rem = p2 + 1;
    tr2.buffer_.insert(tr2.buffer_.end(), tr1.buffer_.begin() + sz_rem, tr1.buffer_.end());
    DCHECK_LE(sz_rem, tr1.buffer_.size());
    tr1.buffer_.resize(sz_rem, tr1.buffer_[0]);  // ugly
    tr1.buc_end_ = tr2.buc_beg_ = split_val;    //更新buc_end_ buc_beg_
    bucket_ranges_[idx_t1 + 1] = split_val;     //更新bucket_ranges_


  } else { // if tree 1 is shorter and move elements from tree 2 to tree 1:
//      DCHECK_GE(sz2, min_elem_split_);
    int to_move_max = int((sz2 - sz1 + imba_diff_ratio_ * sz1) / (2 - imba_diff_ratio_));
    int to_move_mid = int((sz2 - sz1) / 2.0);
    int to_move_min = std::max(0, int((sz2 - sz1 - imba_diff_ratio_ * sz2) / (2 - imba_diff_ratio_)));
    DCHECK_GE(to_move_max, to_move_mid);
    DCHECK_GE(to_move_mid, to_move_min);

    std::vector<int> sort_permu(sz2);
    std::iota(sort_permu.begin(), sort_permu.end(), 0);
    std::sort(sort_permu.begin(), sort_permu.end(), [&](const int &a, const int &b) {
      return tr2.data_tree_[a](bucket_chann_) > tr2.data_tree_[b](bucket_chann_);
    }); // smaller at the back

    int num_to_move = 0;
    KeyFloatType split_val = tr1.buc_end_; // tree1: ( , split val], tree2: [split val, )  想要找到的平衡中值
    if (tr2.data_tree_[sort_permu[sz2 - to_move_mid]](bucket_chann_) !=
        tr2.data_tree_[sort_permu[sz2 - to_move_mid - 1]](bucket_chann_)) { // if no contagious value across split
      num_to_move = to_move_mid;
      split_val = tr2.data_tree_[sort_permu[sz2 - to_move_mid - 1]](bucket_chann_);
    } else {
      KeyFloatType contagious_val = tr2.data_tree_[sort_permu[sz2 - to_move_mid]](bucket_chann_);
      int i = to_move_mid - 1;
      for (; i > to_move_min; i--) {  // look into smaller size
        if (tr2.data_tree_[sort_permu[sz2 - i]](bucket_chann_) != contagious_val) {  // found the split position
          num_to_move = i;
          split_val = contagious_val;
          break;
        }
      }
      if (num_to_move == 0) {
        i = to_move_mid + 1;
        for (; i < to_move_max; i++) {  // look into larger buc side
          if (tr2.data_tree_[sort_permu[sz2 - i]](bucket_chann_) != contagious_val) {  // found the split position
            num_to_move = i - 1;
            split_val = tr2.data_tree_[sort_permu[sz2 - i]](bucket_chann_);
            break;
          }
        }
      }
    }

    if (num_to_move == 0) { // should rebalance but cannot, due to a strip of contagious bucket values
      printf("Cannot split due to contagious values B");
      if (pb1) tr1.popBufferMax(curr_ts);
      tr2.popBufferMax(curr_ts);
      return;
    }

    for (int i = 0; i < num_to_move; i++) {  // add to shorter tree
      tr1.data_tree_.emplace_back(tr2.data_tree_[sort_permu[sz2 - i - 1]]);
      tr1.gkidx_tree_.emplace_back(tr2.gkidx_tree_[sort_permu[sz2 - i - 1]]);
    }

    int p_dat = sz2 - 1, p_perm = sz2 - 1;
    for (; p_perm >= sz2 - num_to_move; p_perm--) {  // rotate to the back;
      while (tr2.data_tree_[p_dat](bucket_chann_) < split_val)
        p_dat--;
      if (sort_permu[p_perm] < p_dat) {
        std::swap(tr2.data_tree_[p_dat], tr2.data_tree_[sort_permu[p_perm]]);
        std::swap(tr2.gkidx_tree_[p_dat], tr2.gkidx_tree_[sort_permu[p_perm]]);
        p_dat--;
      }
    }
    DCHECK_EQ(p_dat + num_to_move, sz2 - 1);
    tr2.data_tree_.resize(p_dat + 1);
    tr2.gkidx_tree_.resize(p_dat + 1, tr2.gkidx_tree_[0]);

    // move buffer: merge and redistribute
    int p1 = 0, p2 = tr2.buffer_.size() - 1, sz_rem;
    while (p1 <= p2) {
      if (tr2.buffer_[p1].pt(bucket_chann_) < split_val && tr2.buffer_[p2].pt(bucket_chann_) >= split_val) {
        std::swap(tr2.buffer_[p1], tr2.buffer_[p2]);
        p1++;
        p2--;
      } else {
        if (tr2.buffer_[p2].pt(bucket_chann_) < split_val)
          p2--;
        if (tr2.buffer_[p1].pt(bucket_chann_) >= split_val)
          p1++;
      }
    }
    sz_rem = p2 + 1;
    tr1.buffer_.insert(tr1.buffer_.end(), tr2.buffer_.begin() + sz_rem, tr2.buffer_.end());
    DCHECK_LE(sz_rem, tr2.buffer_.size());
    tr2.buffer_.resize(sz_rem, tr2.buffer_[0]);
    tr1.buc_end_ = tr2.buc_beg_ = split_val;
    bucket_ranges_[idx_t1 + 1] = split_val;

  }
  //判断data_tree_ buffer_的数据在上下界之内
  for (const auto &dat: tr1.data_tree_) {
    DCHECK_LT(dat(bucket_chann_), tr1.buc_end_);
    DCHECK_GE(dat(bucket_chann_), tr1.buc_beg_);
  }
  for (const auto &dat: tr1.buffer_) {
    DCHECK_LT(dat.pt(bucket_chann_), tr1.buc_end_);
    DCHECK_GE(dat.pt(bucket_chann_), tr1.buc_beg_);
  }

  for (const auto &dat: tr2.data_tree_) {
    DCHECK_LT(dat(bucket_chann_), tr2.buc_end_);
    DCHECK_GE(dat(bucket_chann_), tr2.buc_beg_);
  }
  for (const auto &dat: tr2.buffer_) {
    DCHECK_LT(dat.pt(bucket_chann_), tr2.buc_end_);
    DCHECK_GE(dat.pt(bucket_chann_), tr2.buc_beg_);
  }

  // q: what if all the keys have the same bucket element?
  //  a: calc the number allowed for removal in the larger one, if moving the block will cross the line (another
  //  imba), then do not move.

  // 2. move across boundary

  // 3. rebuild trees  按时间顺序排序buffer_
  std::sort(tr1.buffer_.begin(), tr1.buffer_.end(), [&](const auto &a, const auto &b) {
    return a.ts < b.ts;
  });

  std::sort(tr2.buffer_.begin(), tr2.buffer_.end(), [&](const auto &a, const auto &b) {
    return a.ts < b.ts;
  });

  tr1.popBufferMax(curr_ts);  //全部pop
  tr2.popBufferMax(curr_ts);

}

//层级检索 传入 单层中的某个锚定椭圆的key q_key   候选数量？ k_top   最大限制匹配距离 max_dist_sq   候选id和对应距离容器指针？ res_pairs
/// \param q_key 单层中的某个锚定椭圆的key
/// \param k_top 最大候选数量
/// \param max_dist_sq  最大限制匹配距离
/// \param res_pairs 检索后返回的候选id和对应的检索距离（欧式距离） 已排序，最近的在前
void LayerDB::layerKNNSearch(const RetrievalKey &q_key, const int k_top, const KeyFloatType max_dist_sq,
                             std::vector<std::pair<IndexOfKey, KeyFloatType>> &res_pairs) const {
  // the result vector can have fewer than k result pairs.

  //?这里没有搞明白，为什么要分开多树
  int mid_bucket = 0;
  for (int i = 0; i < max_num_backets_; i++) {
    if (bucket_ranges_[i] <= q_key(bucket_chann_) && bucket_ranges_[i + 1] > q_key(bucket_chann_)) {    //选择bucket_ranges中的与key(bucket_chann_)相近的index
      mid_bucket = i;
      break;
    }
  }

  KeyFloatType max_dist_sq_run = max_dist_sq;


//    std::vector<std::pair<size_t, KeyFloatType>> res_pairs;
  res_pairs.clear();  // all pairs are meaningful, which may fall short of k_top.

  for (int i = 0; i < max_num_backets_; i++) {
    std::vector<IndexOfKey> tmp_gidx;
    std::vector<KeyFloatType> tmp_dists_sq;

    if (i == 0) {
      buckets_[mid_bucket].knnSearch(k_top, tmp_gidx, tmp_dists_sq, q_key, max_dist_sq_run);  //选择mid_bucket层进行匹配
      for (int j = 0; j < k_top; j++)
        if (tmp_dists_sq[j] < max_dist_sq_run)
          res_pairs.emplace_back(tmp_gidx[j], tmp_dists_sq[j]);
        else
          break;

    } else if (mid_bucket - i >= 0) {
      if ((q_key(bucket_chann_) - bucket_ranges_[mid_bucket - i + 1]) *
          (q_key(bucket_chann_) - bucket_ranges_[mid_bucket - i + 1]) > max_dist_sq_run) { continue; }  //超出匹配距离
      buckets_[mid_bucket - i].knnSearch(k_top, tmp_gidx, tmp_dists_sq, q_key, max_dist_sq_run);
      for (int j = 0; j < k_top; j++)
        if (tmp_dists_sq[j] < max_dist_sq_run)
          res_pairs.emplace_back(tmp_gidx[j], tmp_dists_sq[j]);
        else
          break;

    } else if (mid_bucket + i < max_num_backets_) {  // query a
      if ((q_key(bucket_chann_) - bucket_ranges_[mid_bucket + i]) *
          (q_key(bucket_chann_) - bucket_ranges_[mid_bucket + i]) > max_dist_sq_run) { continue; }
      buckets_[mid_bucket + i].knnSearch(k_top, tmp_gidx, tmp_dists_sq, q_key, max_dist_sq_run);
      for (int j = 0; j < k_top; j++)
        if (tmp_dists_sq[j] < max_dist_sq_run)
          res_pairs.emplace_back(tmp_gidx[j], tmp_dists_sq[j]);
        else
          break;

    }
    //对检索出来的候选对进行排序
    std::sort(res_pairs.begin(), res_pairs.end(),
              [&](const std::pair<IndexOfKey, KeyFloatType> &a, const std::pair<IndexOfKey, KeyFloatType> &b) {
                return a.second < b.second;
              });
    if (res_pairs.size() >= k_top) {                  //限制最大的匹配对数量
      res_pairs.resize(k_top, res_pairs[0]);
      max_dist_sq_run = res_pairs.back().second;      //这个max_dist_sq_run 最大限制距离在这更新成当前的最大距离
    }
  }
}

//检索函数 
/// \param num_res 最大返回检索数量
/// \param ret_idx 检索返回的id容器
/// \param out_dist_sq 对应idx的最近邻点的距离值
/// \param q_key 用于检索的键值
/// \param max_dist_sq 限制的最大检索距离？
void TreeBucket::knnSearch(const int num_res, std::vector<IndexOfKey> &ret_idx, std::vector<KeyFloatType> &out_dist_sq,
                           RetrievalKey q_key, const KeyFloatType max_dist_sq) const {
  ret_idx.clear();
  out_dist_sq.resize(num_res);
  std::fill(out_dist_sq.begin(), out_dist_sq.end(), MAX_DIST_SQ);   // in case of fewer points in the tree than k 防止检索到的数量过少，先填充最大值

  if (!tree_ptr) { return; }    //检测不到树

  ret_idx.reserve(num_res);     //分配内存

  std::vector<size_t> idx(num_res, 0);

//    nanoflann::KNNResultSet<KeyFloatType> resultSet(num_res);  // official knn search
//    resultSet.init(&idx[0], &out_dist_sq[0]);

  MyKNNResSet<KeyFloatType> resultSet(num_res);   // knn with max dist
  resultSet.init(&idx[0], &out_dist_sq[0], max_dist_sq);    //输入最大的查询的距离（限制？）

  tree_ptr->index->findNeighbors(resultSet, q_key.data(), nanoflann::SearchParams(10));   //利用key数组进行检索
  for (size_t i = 0; i < num_res; i++) {
    ret_idx.emplace_back(gkidx_tree_[idx[i]]);
  }
}

void TreeBucket::rangeSearch(KeyFloatType worst_dist_sq, std::vector<IndexOfKey> &ret_idx,
                             std::vector<KeyFloatType> &out_dist_sq, RetrievalKey q_key) const {
  std::vector<std::pair<size_t, KeyFloatType>> ret_matches;
  nanoflann::SearchParams params;
  // params.sorted = false;

  ret_idx.clear();
  out_dist_sq.clear();
  if (!tree_ptr) { return; }

  const size_t nMatches = tree_ptr->index->radiusSearch(q_key.data(), worst_dist_sq, ret_matches, params);

  ret_idx.reserve(nMatches);
  out_dist_sq.reserve(nMatches);

  for (size_t i = 0; i < ret_matches.size(); i++) {
    ret_idx.emplace_back(gkidx_tree_[ret_matches[i].first]);
    out_dist_sq.emplace_back(ret_matches[i].second);
  }
}
