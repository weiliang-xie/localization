//
// Created by lewis on 5/6/22.
//
//contour描述符database相关，candidate处理，kn树
#ifndef CONT2_CONTOUR_DB_H
#define CONT2_CONTOUR_DB_H

// #include "contour_mng.h"
#include "contour_mng.h"
#include "correlation.h"
#include "tools/algos.h"

#include <memory>
#include <algorithm>
#include <set>
#include <unordered_set>

#include <nanoflann.hpp>
#include <utility>
#include "KDTreeVectorOfVectorsAdaptor.h"

#include "tools/bm_util.h"

extern SequentialTimeProfiler stp;

//typedef Eigen::Matrix<KeyFloatType, 4, 1> tree_key_t;
typedef std::vector<RetrievalKey> my_vector_of_vectors_t;
typedef KDTreeVectorOfVectorsAdaptor<my_vector_of_vectors_t, KeyFloatType> my_kd_tree_t;

const KeyFloatType MAX_BUCKET_VAL = 1000.0f;
const KeyFloatType MAX_DIST_SQ = 1e6;

template<
    typename _DistanceType, typename _IndexType = size_t,
    typename _CountType = size_t>
class MyKNNResSet : public nanoflann::KNNResultSet<_DistanceType, _IndexType, _CountType> {
public:
  using DistanceType = _DistanceType;
  using IndexType = _IndexType;
  using CountType = _CountType;

  inline explicit MyKNNResSet(CountType capacity_)
      : nanoflann::KNNResultSet<_DistanceType, _IndexType, _CountType>(capacity_) {
  }

  inline void init(IndexType *indices_, DistanceType *dists_, DistanceType max_dist_metric) {
    this->indices = indices_;
    this->dists = dists_;
    this->count = 0;
    if (this->capacity)
      this->dists[this->capacity - 1] = max_dist_metric;
  }
};

//添加到树的最大空间/时间延迟，等待的最小空间/时间延迟
struct TreeBucketConfig {
  double max_elapse_ = 25.0;  // the max spatial/temporal delay before adding to the trees
  double min_elapse_ = 15.0;  // the min spatial/temporal delay to wait before adding to the trees
};


struct IndexOfKey {  // where does the key come from? 1) global idx, 2) level, 3) ith/seq at that level //key的位置序号
  size_t gidx{};  //global idx 帧级序号
  int level{};    //level  层序
  int seq{};      //ith/seq at that level 在层级上的序号

  IndexOfKey(size_t g, int l, int s) : gidx(g), level(l), seq(s) {}
};

//! The minimal unit of a tree/wrapper of a kd-tree
struct TreeBucket {

  struct RetrTriplet {  // retrieval triplet 检索三元组
    RetrievalKey pt;    //检索的数据 key值数组
    double ts{};
    IndexOfKey iok;   //key的索引

//    RetrTriplet() = default;

    RetrTriplet(const RetrievalKey &_a, double _b, size_t g, int l, int s) : pt(_a), ts(_b), iok(g, l, s) {}

    RetrTriplet(const RetrievalKey &_a, double _b, IndexOfKey i) : pt(_a), ts(_b), iok(i) {}
  };

  const TreeBucketConfig cfg_;

  KeyFloatType buc_beg_{}, buc_end_{};  // [beg, end)
  my_vector_of_vectors_t data_tree_;    //数据类型与key一致 存储key
  std::shared_ptr<my_kd_tree_t> tree_ptr = nullptr;
  std::vector<RetrTriplet> buffer_;    // ordered, ascending 在bucket内
  std::vector<IndexOfKey> gkidx_tree_;  // global index of LECDManager in the whole database   //数据库中存储的key对应的位置信息

  //输入 配置值，[beg, end)
  TreeBucket(const TreeBucketConfig &config, KeyFloatType beg, KeyFloatType end) : cfg_(config), buc_beg_(beg),
                                                                                   buc_end_(end) {}

  //?返回当前bucket中data_tree_的容量（?当前所有点云帧数量）
  size_t getTreeSize() const {
    DCHECK_EQ(data_tree_.size(), gkidx_tree_.size());     //data_tree_数量与当前所有点云帧数量是否相等？
    return data_tree_.size();
  }

  void pushBuffer(const RetrievalKey &tree_key, double ts, IndexOfKey iok) {
    buffer_.emplace_back(tree_key, ts, iok);
  }

  //通过buffer内的帧时间是否足够远来判断是否有需要pop的buffer 
  inline bool needPopBuffer(double curr_ts) const {
    double ts_overflow = curr_ts - cfg_.max_elapse_;
    if (buffer_.empty() || buffer_[0].ts > ts_overflow)  // rebuild every (max-min) sec, ignore newest min. buffer内最早时间仍在当前时间的前max_elapse_内，无需pop
      return false;
    return true;
  }

  inline void rebuildTree() {
    if (tree_ptr)
      // is this an efficient rebuild when number of elements change?
      // should be OK, since the data array is stored in flann as const alias, will not be affected by reassigning.
      tree_ptr->index->buildIndex();    //? 这里如何在已有指针的前提下重建？
    else
      tree_ptr = std::make_shared<my_kd_tree_t>(RetrievalKey::SizeAtCompileTime /*dim*/, data_tree_,
                                                10 /* max leaf */ );
  }

  /// Pop max possible from the buffer into the tree, and rebuild the tree 传入缓冲区内距离当前时间min_elapse_之前的数据并重建树
  /// \param curr_ts
  void popBufferMax(double curr_ts) {
    double ts_cutoff = curr_ts - cfg_.min_elapse_;
    int gap = 0;
    for (; gap < buffer_.size(); gap++) {
      if (buffer_[gap].ts >= ts_cutoff) {
        break;
      }
    }

    if (gap > 0) {  //没有则
      size_t sz0 = data_tree_.size();
      DCHECK_EQ(sz0, gkidx_tree_.size());
      data_tree_.reserve(sz0 + gap);
      gkidx_tree_.reserve(sz0 + gap);   //扩容
      for (size_t i = 0; i < gap; i++) {
        data_tree_.emplace_back(buffer_[i].pt);
        gkidx_tree_.emplace_back(buffer_[i].iok);
      }
      buffer_.erase(buffer_.begin(), buffer_.begin() + gap);    //删除已经存入的缓冲区buffer

      rebuildTree();    //重建
    }
  }

  ///
  /// \param num_res
  /// \param ret_idx
  /// \param out_dist_sq The size must be num_res
  /// \param q_key
  void knnSearch(const int num_res, std::vector<IndexOfKey> &ret_idx, std::vector<KeyFloatType> &out_dist_sq,
                 RetrievalKey q_key, const KeyFloatType max_dist_sq) const;

  void rangeSearch(KeyFloatType worst_dist_sq, std::vector<IndexOfKey> &ret_idx, std::vector<KeyFloatType> &out_dist_sq,
                   RetrievalKey q_key) const;

};

//! The manager of a bucket of kd-trees at a layer
struct LayerDB {
  static const int min_elem_split_ = 100;
  static constexpr double imba_diff_ratio_ = 0.2; // if abs(size(i)-size(i+1))>ratio * max(,), we need to balance the two trees. //?平衡树是什么意思
  static const int max_num_backets_ = 6;  // number of trees per layer  一层一树？
  static const int bucket_chann_ = 0; // the #th dimension of the retrieval key that is used as buckets.  用来检索bucket的key值维度（第几个key）

  std::vector<TreeBucket> buckets_;          //构建则初始化 第一个的beg为-MAX_BUCKET_VAL, 后续则全为MAX_BUCKET_VAL 初始化后的数量为max_num_backets_
  std::vector<KeyFloatType> bucket_ranges_;  // [min, max) pairs for buckets' range bucket的分割临界值 构建则初始化 size为max_num_backets_ + 1 前面第一填入-MAX_BUCKET_VAL，后续为MAX_BUCKET_VAL
  
  //初始化？这里是将KeyFloatType容器的前面第一填入-MAX_BUCKET_VAL，后续为MAX_BUCKET_VAL，同时填充TreeBucket
  explicit LayerDB(const TreeBucketConfig &tb_cfg) {
    bucket_ranges_.resize(max_num_backets_ + 1);
    bucket_ranges_.front() = -MAX_BUCKET_VAL;
    bucket_ranges_.back() = MAX_BUCKET_VAL;
    buckets_.emplace_back(tb_cfg, -MAX_BUCKET_VAL, MAX_BUCKET_VAL);

    // empty buckets
    for (int i = 1; i < max_num_backets_; i++) {
      bucket_ranges_[i] = MAX_BUCKET_VAL;
      buckets_.emplace_back(tb_cfg, MAX_BUCKET_VAL, MAX_BUCKET_VAL);
    }
  }

  //重载构造函数，如有现成的，直接传入
  LayerDB(const LayerDB &ldb) : buckets_(ldb.buckets_), bucket_ranges_(ldb.bucket_ranges_) {}

  // add buffer 将数据和索引存入TreeBucket 容器  依据key[0]在bucket_ranges_中寻找区间，并存入对应区间的buckets_
  /// \param layer_key 需要保存的key值
  /// \param layer_key 保存的帧的时间戳
  /// \param layer_key key的序号信息
  void pushBuffer(const RetrievalKey &layer_key, double ts, IndexOfKey scan_key_gidx) {
    for (int i = 0; i < max_num_backets_; i++) {
      if (bucket_ranges_[i] <= layer_key(bucket_chann_) && layer_key(bucket_chann_) < bucket_ranges_[i + 1]) {    //?依据key[0]在bucket_ranges_中寻找区间，并存入对应区间的buckets_
        if (layer_key.sum() != 0) // if an all zero key, we do not add it.
          buckets_[i].pushBuffer(layer_key, ts, scan_key_gidx);
        return;
      }
    }
  }

  void rebuild(int idx_t1, double curr_ts);

  //层级检索 传入 单层中的某个锚定椭圆的key q_key   候选数量？ k_top   最大限制匹配距离 max_dist_sq   候选id和对应距离容器指针？ res_pairs
/// \param q_key 单层中的某个锚定椭圆的key
/// \param k_top 最大候选数量
/// \param max_dist_sq  最大限制匹配距离
/// \param res_pairs 检索后返回的候选id和对应的检索距离（欧式距离） 已排序，最近的在前
  void layerKNNSearch(const RetrievalKey &q_key, const int k_top, const KeyFloatType max_dist_sq,
                      std::vector<std::pair<IndexOfKey, KeyFloatType>> &res_pairs) const;

  // TODO:
  void layerRangeSearch(const RetrievalKey &q_key, const KeyFloatType max_dist_sq,
                        std::vector<std::pair<IndexOfKey, KeyFloatType>> &res_pairs) const {
    res_pairs.clear();
    for (int i = 0; i < max_num_backets_; i++) {
      std::vector<IndexOfKey> tmp_gkidx;
      std::vector<KeyFloatType> tmp_dists_sq;
      buckets_[i].rangeSearch(max_dist_sq, tmp_gkidx, tmp_dists_sq, q_key);
      for (int j = 0; j < tmp_gkidx.size(); j++) {
        res_pairs.emplace_back(tmp_gkidx[j], tmp_dists_sq[j]);
      }
    }
  }

};

//struct CandSimScore {
//  // Our similarity score is multi-dimensional
//  int cnt_constell_ = 0;
//  int cnt_pairwise_sim_ = 0;
//  double correlation_ = 0;
//
//  CandSimScore() = default;
//
//  CandSimScore(int cnt_chk1, int cnt_chk2, double init_corr) : cnt_constell_(cnt_chk1), cnt_pairwise_sim_(cnt_chk2),
//                                                               correlation_(init_corr) {}
//
//  bool operator<(const CandSimScore &a) const {
//    if (cnt_pairwise_sim_ == a.cnt_pairwise_sim_)
//      return correlation_ < a.correlation_;
//    return cnt_pairwise_sim_ < a.cnt_pairwise_sim_;
//  }
//
//  bool operator>(const CandSimScore &a) const {
//    if (cnt_pairwise_sim_ == a.cnt_pairwise_sim_)
//      return correlation_ > a.correlation_;
//    return cnt_pairwise_sim_ > a.cnt_pairwise_sim_;
//  }
//};

// A set of all the check parameters combined as a whole 这是一个lecd匹配后的得分集合
struct CandidateScoreEnsemble {
  ScoreConstellSim sim_constell;  //结构群相似度
  ScorePairwiseSim sim_pair;      //相似匹配对
  ScorePostProc sim_post;

};

//TODO 待完善介绍
struct CandidateManager {

  struct CandidateAnchorProp {
    std::map<ConstellationPair, float> constell_;  // map of {constellation matches: percentage score}    //map of 匹配对和匹配对的网格数量占比
    Eigen::Isometry2d T_delta_;  // the key feature that distinguishes different proposals 区别不同提案的关键特征 这里指的应该是帧与帧之间相似匹配对的相对位姿(anchor + peripheral)均值(在image坐标系下)
    float correlation_ = 0;
    int vote_cnt_ = 0;  // cnt of matched lecds voting for this TF (not cnt of unique pairs) //具有相同位姿的匹配对数量
    float area_perc_ = 0;  // a weighted sum of the area % of used lecds in all the lecds at different levels.按权重相加各key层的面积百分比和
    // TODO: should we record area percentage as the metric for "votes"?
  };

  //TODO candidate使用 点云帧为单位，存储后续优化过滤相关的数据（最相似的3对锚定群匹配对容）
  struct CandidatePoseData {
    std::shared_ptr<const LECDManager> cm_cand_;   //匹配到的候选描述符指针
    std::unique_ptr<ConstellCorrelation> corr_est_;  // generate the correlation estimator after polling all the cands  匹配生成的估计相关性类指针 不通过则为空
    std::vector<CandidateAnchorProp> anch_props_;     //存储相似的相对位姿的均值，容器内仅限3个

    /// \param T_prop         //src 和 tgt的相对位姿
    /// \param sim_pairs      // 完成过滤的匹配对
    /// \param sim_area_perc The level percentage score of a corresponding constellation  //完成过滤的匹配对对应的网格数量占比
    void addProposal(const Eigen::Isometry2d &T_prop, const std::vector<ConstellationPair> &sim_pairs,
                     const std::vector<float> &sim_area_perc) {
      DCHECK_GT(sim_pairs.size(), 3);  // hard bottom line  保证至少有3对以上的匹配对
      DCHECK_EQ(sim_pairs.size(), sim_area_perc.size());

      //遍历已有的anchor props 寻找接近的位姿特征并添加上去
      for (int i = 0; i < anch_props_.size(); i++) {
        const Eigen::Isometry2d delta_T = T_prop.inverse() * anch_props_[i].T_delta_;     //计算估计的初始位姿与这个anchor的相对位姿误差
        // hardcoded threshold: 2.0m, 0.3.rad
        if (delta_T.translation().norm() < 2.0 && std::abs(std::atan2(delta_T(1, 0), delta_T(0, 0))) < 0.3) { //这里判断差异的旋转角和平移距离 误差满足阈值才能添加上去
          for (int j = 0; j < sim_pairs.size(); j++) {
            anch_props_[i].constell_.insert({sim_pairs[j], sim_area_perc[j]});  // unique 将匹配对和网格数量占比拆开一一填入
          }

          anch_props_[i].vote_cnt_ += sim_pairs.size();  // not unique    //求数量总和
          int w1 = anch_props_[i].vote_cnt_, w2 = sim_pairs.size();
          Eigen::Vector2d trans_bl =
              (anch_props_[i].T_delta_.translation() * w1 + T_prop.translation() * w2) / (w1 + w2);     //求原有的和新加入的平移向量的平均值
          double ang1 = std::atan2(anch_props_[i].T_delta_(1, 0), anch_props_[i].T_delta_(0, 0));
          double ang2 = std::atan2(T_prop(1, 0), T_prop(0, 0));

          //这部分是什么？
          double diff = ang2 - ang1;
          if (diff < 0) diff += 2 * M_PI;
          if (diff > M_PI) diff -= 2 * M_PI;
          double ang_bl = diff * w2 / (w1 + w2) + ang1;     //将新加入的差值均化后加入到原有差值中 计算加权平均后差值并返回至原有的

          anch_props_[i].T_delta_.setIdentity();
          anch_props_[i].T_delta_.rotate(ang_bl);  // no need to clamp
          anch_props_[i].T_delta_.pretranslate(trans_bl);

          return;  // greedy
        }
      }

      if (anch_props_.size() > 3)     //只能有三个props吗？ 目的是将差异大的也排除在外？不进行记录
        return; // limit the number of different proposals w.r.t. a pose

      anch_props_.emplace_back();
      anch_props_.back().T_delta_ = T_prop;
      for (int j = 0; j < sim_pairs.size(); j++) {
        anch_props_.back().constell_.insert({sim_pairs[j], sim_area_perc[j]});  // unique
      }
      anch_props_.back().vote_cnt_ = sim_pairs.size();

    }
  };

  //=============================================================

//  const CandSimScore score_lb_;  // score lower/upper bound
  std::shared_ptr<const LECDManager> cm_tgt_;    //需要查询的描述符指针

  // dynamic thresholds param and var for different checks. Used to replace `score_lb_`
  const CandidateScoreEnsemble sim_ub_;  // the upper bound of check thres 检测阈值上界
  CandidateScoreEnsemble sim_var_;   // the (dynamic) lower bound of check thres, increase with positive predictions 检测阈值下界（动态？）

  // data structure
  std::map<int, int> cand_id_pos_pair_;       //key：匹配到的候选点云帧的全局i data：匹配帧在CandidatePoseData容器中的位置值
  std::vector<CandidatePoseData> candidates_; //通过筛选的候选数据容器

  int flow_valve = 0; // avoid to reverse the work flow 避免颠倒工作流程？
  int cand_aft_check1 = 0;  // number of candidate occurrence (not unique places) after each check 经过每个检查后剩下的候选数
  int cand_aft_check2 = 0;
  int cand_aft_check3 = 0;

  //构造函数，传入需要查询的单帧描述符指针，阈值的上界和下界
  CandidateManager(std::shared_ptr<const LECDManager> cm_q,
                   const CandidateScoreEnsemble sim_lb, const CandidateScoreEnsemble sim_ub) :
      cm_tgt_(std::move(cm_q)), sim_var_(sim_lb), sim_ub_(sim_ub) {
    // TO-DO: pass in sim score lb and ub as params
    CHECK(sim_lb.sim_constell.strictSmaller(sim_ub.sim_constell));
    CHECK(sim_lb.sim_pair.strictSmaller(sim_ub.sim_pair));
    CHECK(sim_lb.sim_post.strictSmaller(sim_ub.sim_post));
  }

  /// Main func 1/3: check possible anchor pairing and add to the database 过滤可能的锚点配对并将匹配到的添加到数据库 1.统计数据相似度 2.BCI匹配 3.外围椭圆的统数和长轴投影匹配
  /// \param cm_cand lecd manager for the candidate  候选lecd
  /// \param anchor_pair "hint": the anchor for key and bci pairing  //匹配好的候选和查询anchor lecd对结构
  /// \param lecd_sim 匹配相似度相关阈值
  /// \return
  CandidateScoreEnsemble checkCandWithHint(const std::shared_ptr<const LECDManager> &cm_cand,
                                           const ConstellationPair &anchor_pair,
                                           const LECDSimThresConfig &lecd_sim) {
    DCHECK(flow_valve == 0);
    int cand_id = cm_cand->getIntID();
    CandidateScoreEnsemble ret_score;

    // check: (1/4) anchor similarity 返回统计数据是否吻合
    bool anchor_sim = LECDManager::checkContPairSim(*cm_cand, *cm_tgt_, anchor_pair, lecd_sim);
    if (!anchor_sim)
      return ret_score;
    cand_aft_check1++;

    // check (2/4): pure constellation check
    std::vector<ConstellationPair> tmp_pairs1;
    ScoreConstellSim ret_constell_sim = BCI::checkConstellSim(cm_cand->getBCI(anchor_pair.level, anchor_pair.seq_src),
                                                              cm_tgt_->getBCI(anchor_pair.level, anchor_pair.seq_tgt),
                                                              sim_var_.sim_constell, tmp_pairs1);
    ret_score.sim_constell = ret_constell_sim;
    if (ret_constell_sim.overall() < sim_var_.sim_constell.overall())   //最长子序列长度不过关，直接返回得分结果
      return ret_score;
    cand_aft_check2++;

    // check (3/4): individual similarity check
    std::vector<ConstellationPair> tmp_pairs2;    //完成checkCandWithHint匹配的匹配对
    std::vector<float> tmp_area_perc;             //完成checkCandWithHint匹配的匹配对对应的网格数量占比
    ScorePairwiseSim ret_pairwise_sim = LECDManager::checkConstellCorrespSim(*cm_cand, *cm_tgt_, tmp_pairs1,
                                                                                sim_var_.sim_pair, lecd_sim,
                                                                                tmp_pairs2, tmp_area_perc);
    ret_score.sim_pair = ret_pairwise_sim;
    if (ret_pairwise_sim.overall() < sim_var_.sim_pair.overall())   //外围椭圆匹配过滤后的匹配对数量不够，直接返回结果
      return ret_score;
     cand_aft_check3++;

    // 2. Get the transform between the two scans 获取相对位姿
    Eigen::Isometry2d T_pass = LECDManager::getTFFromConstell(*cm_cand, *cm_tgt_, tmp_pairs2.begin(),
                                                                 tmp_pairs2.end());

    // 3. add to/update candidates_  往candidates_ 和 cand_id_pos_pair_ 添加匹配点云帧数据
    auto cand_it = cand_id_pos_pair_.find(cand_id);
    if (cand_it != cand_id_pos_pair_.end()) {
      // the candidate pose exists
      candidates_[cand_it->second].addProposal(T_pass, tmp_pairs2, tmp_area_perc);
    } else {
      // add new
      CandidatePoseData new_cand;
      new_cand.cm_cand_ = cm_cand;
      new_cand.addProposal(T_pass, tmp_pairs2, tmp_area_perc);
      cand_id_pos_pair_.insert({cand_id, candidates_.size()});
      candidates_.emplace_back(std::move(new_cand));
    }

    // correlation calculation
    // TODO: merge results for the same candidate pose

    return ret_score;
  }

  // pre-calculate the correlation scores for each candidate set, and check the correlation scores. 预先计算每个候选集合的相关性得分，并检查相关性得分
  /// Main func 2/3:
  //整理候选 整理的候选数据来自于candidates_   key层面积均值过小 ||  相对位姿平移距离过大 || 初始位姿相关度过小 都剔除
  void tidyUpCandidates() {
    DCHECK(flow_valve < 1);   //流程标志
    flow_valve++;
    GMMOptConfig gmm_config;
    // printf("Tidy up pose %lu candidates.\n", candidates_.size());

    int cnt_to_rm = 0;

    // analyze the anchor pairs for each pose 遍历候选
    for (auto &candidate: candidates_) {
      DCHECK(!candidate.anch_props_.empty());
      // find the best T_init for setting up correlation problem estimation (based on vote)
      int idx_sel = 0;  //匹配对最多的候选序号
      for (int i = 0; i < candidate.anch_props_.size(); i++) {  // TODO: should we use vote count or area?

        // get the percentage of points used in match v. total area of each level.
        std::vector<float> lev_perc(cm_tgt_->getConfig().lv_grads_.size(), 0);    //容量为层数
        for (const auto &pr: candidate.anch_props_[i].constell_) {
          lev_perc[pr.first.level] += pr.second;    //当前帧完成匹配通过筛选的网格数量占比

        }

        float perc = 0;
        for (int j = 0; j < NUM_BIN_KEY_LAYER; j++)     //key 层层数
          perc += LAYER_AREA_WEIGHTS[j] * lev_perc[DIST_BIN_LAYERS[j]];   //按权重相加key层的面积百分比

        candidate.anch_props_[i].area_perc_ = perc;

        if (candidate.anch_props_[i].vote_cnt_ > candidate.anch_props_[idx_sel].vote_cnt_)
          idx_sel = i;
      }
      // put the best prop to the first and leave the rest as they are 将匹配对最多的候选位姿放到最前
      std::swap(candidate.anch_props_[0], candidate.anch_props_[idx_sel]);

      if (candidate.anch_props_[0].area_perc_ < sim_var_.sim_post.area_perc) { // check (1/3): area score.    //?过小的不采纳？ //面积占比是否符合条件
        cnt_to_rm++;
        continue;
      }

      // Overall test 2: Censor distance. NOTE: The negate!! Larger is better
      double neg_est_trans_norm2d = -ConstellCorrelation::getEstSensTF(candidate.anch_props_[0].T_delta_,
                                                                       cm_tgt_->getConfig()).translation().norm();
      if (neg_est_trans_norm2d < sim_var_.sim_post.neg_est_dist) { // check (2/3): area score.    //位姿平移距离过大忽略
        cnt_to_rm++;
        continue;
      }

      std::unique_ptr<ConstellCorrelation> corr_est(new ConstellCorrelation(gmm_config));
      auto corr_score_init = (float) corr_est->initProblem(*(candidate.cm_cand_), *cm_tgt_,
                                                           candidate.anch_props_[0].T_delta_);

      if (corr_score_init < sim_var_.sim_post.correlation) { // check (3/3): correlation score.  //相关数低于阈值不考虑
        cnt_to_rm++;
        continue;
      }

      candidate.corr_est_ = std::move(corr_est);
    }

    // remove poses failing the corr check 给候选容器内的候选排序，估计指针corr_est_非空的在前
    int p1 = 0, p2 = candidates_.size() - 1;
    while (p1 <= p2) {
      if (!candidates_[p1].corr_est_ && candidates_[p2].corr_est_) {
        std::swap(candidates_[p1], candidates_[p2]);
        p1++;
        p2--;
      } else {
        if (candidates_[p1].corr_est_) p1++;
        if (!candidates_[p2].corr_est_) p2--;
      }
    }
    CHECK_EQ(p2 + 1 + cnt_to_rm, candidates_.size());   //判断总数
    candidates_.erase(candidates_.begin() + p2 + 1, candidates_.end()); //删除不通过估计的候选

  }

  /// Main func 3/3: pre select hopeful pose candidates, and optimize for finer pose estimations.
  /// 在候选中提取限制最大数量的相关性最大的帧，并优化二维变换矩阵，只返回1个最优值
  /// \param max_fine_opt 最大的优化相对位姿相关度对比数量
  /// \param res_cand 选取的候选描述符指针
  /// \param res_corr 选取的候选的相关度
  /// \param res_T 选取的候选的优化后相对位姿
  /// \return  //最终选取的候选数量
  int fineOptimize(int max_fine_opt, std::vector<std::shared_ptr<const LECDManager>> &res_cand,
                   std::vector<double> &res_corr, std::vector<Eigen::Isometry2d> &res_T) {
    DCHECK(flow_valve < 2);
    flow_valve++;

    res_cand.clear();
    res_corr.clear();
    res_T.clear();

    if (candidates_.empty())  //没有候选，直接返回
      return 0;

    //!依据候选中的最大相关度进行排序 这个相关度在前面一直都没有被处理过 似乎一直是0值
    std::sort(candidates_.begin(), candidates_.end(), [&](const CandidatePoseData &d1, const CandidatePoseData &d2) {
      return d1.anch_props_[0].correlation_ > d2.anch_props_[0].correlation_;
    });

    int pre_sel_size = std::min(max_fine_opt, (int) candidates_.size());      //确定最大可提取候选值
    for (int i = 0; i < pre_sel_size; i++) {
      auto tmp_res = candidates_[i].corr_est_->calcCorrelation();  // fine optimize 优化位姿
      candidates_[i].anch_props_[0].correlation_ = tmp_res.first;   //更新相关度
      candidates_[i].anch_props_[0].T_delta_ = tmp_res.second;      //更新位姿
    }

    //重新依据相关度从大到小排序 candidates_中前"最大可提取候选数量"个候选
    std::sort(candidates_.begin(), candidates_.begin() + pre_sel_size,
              [&](const CandidatePoseData &d1, const CandidatePoseData &d2) {
                return d1.anch_props_[0].correlation_ > d2.anch_props_[0].correlation_;
              });

    int ret_size = 1;  // the needed 需要的最终候选
    for (int i = 0; i < ret_size; i++) {
      res_cand.emplace_back(candidates_[i].cm_cand_);
      res_corr.emplace_back(candidates_[i].anch_props_[0].correlation_);
      res_T.emplace_back(candidates_[i].anch_props_[0].T_delta_);
    }

    return ret_size;
  }
  void selfCensor() {

  }

};

struct LECDDBConfig {
  int nnk_ = 50;  //
  int max_fine_opt_ = 10;
  std::vector<int> q_levels_;  // the layers to generate anchors (Note the difference between `DIST_BIN_LAYERS`) 生成锚定的层 [ 1, 2, 3 ]

  LECDSimThresConfig lecd_sim_cfg_;      //lecd匹配对的相似度阈值，从参数文件中获取
  TreeBucketConfig tb_cfg_;
};

// manages the whole database of lecds for place re-identification
// top level database
class LECDDB {
  const LECDDBConfig cfg_;

  std::vector<LayerDB> layer_db_;   //层级上的数据库
  std::vector<std::shared_ptr<const LECDManager>> all_bevs_;   //存放所有点云帧的lecdmanager指针的容器

public:
  LECDDB(const LECDDBConfig &config) : cfg_(config) {
    for (auto i: cfg_.q_levels_)
      layer_db_.emplace_back(TreeBucketConfig(cfg_.tb_cfg_));
    CHECK(!cfg_.q_levels_.empty());
  }

  // TODO: unlike queryKNN, this one directly calculates relative transform and requires no post processing
  //  outside the function. The returned cmng are the matched ones.
  //查询检索获取候选 是最终确定的唯一（目前）候选
  /// \param q_ptr   待查询的描述符指针
  /// \param thres_lb  阈值上界
  /// \param thres_ub  阈值下界
  /// \param cand_ptrs 最终候选的描述符指针
  /// \param cand_corr 最终候选的相关度
  /// \param cand_tf candidates are src/old, T_tgt = T_delta * T_src 最终的候选相对位姿
  int queryRangedKNN(const std::shared_ptr<const LECDManager> &q_ptr,
                      const CandidateScoreEnsemble &thres_lb,
                      const CandidateScoreEnsemble &thres_ub,
                      std::vector<std::shared_ptr<const LECDManager>> &cand_ptrs,
                      std::vector<double> &cand_corr,
                      std::vector<Eigen::Isometry2d> &cand_tf) const {
    cand_ptrs.clear();
    cand_corr.clear();
    cand_tf.clear();

    double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0;
    TicToc clk;
    CandidateManager cand_mng(q_ptr, thres_lb, thres_ub);

    //每一anchor层的每一个anchor lecd都会进行检索匹配，并作为匹配对保存
    for (int ll = 0; ll < cfg_.q_levels_.size(); ll++) {    //这里的不是完整的层数
      const std::vector<BCI> &q_bcis = q_ptr->getLevBCI(cfg_.q_levels_[ll]);
      std::vector<RetrievalKey> q_keys = q_ptr->getLevRetrievalKey(cfg_.q_levels_[ll]);
      DCHECK_EQ(q_bcis.size(), q_keys.size());
      for (int seq = 0; seq < q_bcis.size(); seq++) {   //遍历锚定椭圆数量
        if (q_keys[seq].sum() != 0) {                   //判断是否有key
          // 1. query
          clk.tic();
          stp.start();
          std::vector<std::pair<IndexOfKey, KeyFloatType>> tmp_res;
          KeyFloatType key_bounds[3][2];
          key_bounds[0][0] = q_keys[seq][0] * 0.8;  // sqrt(max_eig*cnt)
          key_bounds[0][1] = q_keys[seq][0] / 0.8;

          key_bounds[1][0] = q_keys[seq][1] * 0.8;  // sqrt(min_eig*cnt)
          key_bounds[1][1] = q_keys[seq][1] / 0.8;

          key_bounds[2][0] = q_keys[seq][2] * 0.8 * 0.75;  // com*cnt
          key_bounds[2][1] = q_keys[seq][2] / (0.8 * 0.75);

          KeyFloatType dist_ub = 1e6;
          //这里想要获取的是什么的最大值的和？
          dist_ub = std::max((q_keys[seq][0] - key_bounds[0][0]) * (q_keys[seq][0] - key_bounds[0][0]),
                             (q_keys[seq][0] - key_bounds[0][1]) * (q_keys[seq][0] - key_bounds[0][1]))
                    + std::max((q_keys[seq][1] - key_bounds[1][0]) * (q_keys[seq][1] - key_bounds[1][0]),
                               (q_keys[seq][1] - key_bounds[1][1]) * (q_keys[seq][1] - key_bounds[1][1]))
                    + std::max((q_keys[seq][2] - key_bounds[2][0]) * (q_keys[seq][2] - key_bounds[2][0]),
                               (q_keys[seq][2] - key_bounds[2][1]) * (q_keys[seq][2] - key_bounds[2][1]));

          layer_db_[ll].layerKNNSearch(q_keys[seq], cfg_.nnk_, dist_ub, tmp_res);     //利用当前anchor的key值进行检索，更新检索返回anchor的id-dist对：tmp_res
          stp.record("KNN search");
          t1 += clk.toc();

          // 2. check
          stp.start();
          //遍历检索返回的数据  50对匹配对 通过筛选的生成候选
          for (const auto &sear_res: tmp_res) {
            clk.tic();
            auto cnt_chk_pass = cand_mng.checkCandWithHint(all_bevs_[sear_res.first.gidx],
                                                           ConstellationPair(cfg_.q_levels_[ll], sear_res.first.seq,
                                                                             seq),
                                                           cfg_.lecd_sim_cfg_);     //过滤匹配对，1.统计数据相似度 2.BCI匹配 3.外围椭圆的统数和长轴投影匹配
            t2 += clk.toc();
          }
          stp.record("Constell");

        }
      }
    }

    // find the best ones with fine-tuning:  筛选出最好的匹配对
    std::vector<std::shared_ptr<const LECDManager>> res_cand_ptr;
    std::vector<double> res_corr;
    std::vector<Eigen::Isometry2d> res_T;

    clk.tic();
    stp.start();
    cand_mng.tidyUpCandidates();
    int num_best_cands = cand_mng.fineOptimize(cfg_.max_fine_opt_, res_cand_ptr, res_corr, res_T);
    stp.record("L2 opt");
    t5 += clk.toc();

    for (int i = 0; i < num_best_cands; i++) {
      cand_ptrs.emplace_back(res_cand_ptr[i]);
      cand_corr.emplace_back(res_corr[i]);
      cand_tf.emplace_back(res_T[i]);
    }

    return num_best_cands;
    // TODO: separate pose refinement into another protected function
  }

  // TO-DO: 2. add a scan, and retrieval data to buffer
  //保存描述符数据
  void addScan(const std::shared_ptr<LECDManager> &added, double curr_timestamp) {
    for (int ll = 0; ll < cfg_.q_levels_.size(); ll++) {
      int seq = 0; // key seq in a layer for a given scan.
      for (const auto &permu_key: added->getLevRetrievalKey(cfg_.q_levels_[ll])) {
        if (permu_key.sum() != 0)
          layer_db_[ll].pushBuffer(permu_key, curr_timestamp, IndexOfKey(all_bevs_.size(), cfg_.q_levels_[ll], seq));
        seq++;
      }
    }
    all_bevs_.emplace_back(added);
  }

  // TO-DO: 3. push data popped from buffer, and maintain balance (at most 2 buckets at a time)
  //将描述符放入检索树 并重建树 输入 seed：新加入的点云帧序号  curr_timestamp：当前时间戳
  void pushAndBalance(int seed, double curr_timestamp) {
    int idx_t1 = std::abs(seed) % (2 * (layer_db_[0].max_num_backets_ - 2));
    if (idx_t1 > (layer_db_[0].max_num_backets_ - 2))
      idx_t1 = 2 * (layer_db_[0].max_num_backets_ - 2) - idx_t1;    //?求反值 0~4 与点云帧序号有关

    // printf("Balancing bucket %d and %d\n", idx_t1, idx_t1 + 1);

    // printf("Tree size of each bucket: \n");
    for (int ll = 0; ll < cfg_.q_levels_.size(); ll++) {
      layer_db_[ll].rebuild(idx_t1, curr_timestamp);
    }
  }

};


#endif //CONT2_CONTOUR_DB_H
