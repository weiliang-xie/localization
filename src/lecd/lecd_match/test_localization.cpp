

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
// #include <matplotlibcpp.h>

// namespace plt = matplotlibcpp;

const std::string PROJ_DIR = std::string(PJSRCDIR);

SequentialTimeProfiler stp;

//这个继承自spinner_ros.h(该头文件提供用于订阅话题并发送相关话题的函数) 还包含位姿真值的结构体
class BatchBinSpinner : public BaseROSSpinner {
public:
  // --- Added members for evaluation and LC module running ---
  std::unique_ptr<ContourDB> ptr_contour_db;          //描述符数据库指针
  std::unique_ptr<ContLCDEvaluator> ptr_evaluator;    //评价部分指针

  ContourManagerConfig cm_config;
  ContourDBConfig db_config;

  CandidateScoreEnsemble thres_lb_, thres_ub_;  // check thresholds variable query parameters 查询阈值 分上界下界

  // bookkeeping
  int cnt_tp = 0, cnt_fn = 0, cnt_fp = 0;
  double ts_beg = -1;

  //构造函数传入ros句柄
  explicit BatchBinSpinner(ros::NodeHandle &nh_) : BaseROSSpinner(nh_) {  // mf1 k02

  }

  // before start: 1/1: load thres
  //  
  void loadConfig(const std::string &config_fpath, std::string &sav_path) {

    printf("Loading parameters...\n");
    auto yl = yamlLoader(config_fpath);

    std::string fpath_sens_gt_pose, fpath_lidar_bins;
    double corr_thres;

    yl.loadOneConfig({"fpath_sens_gt_pose"}, fpath_sens_gt_pose);
    yl.loadOneConfig({"fpath_lidar_bins"}, fpath_lidar_bins);
    yl.loadOneConfig({"correlation_thres"}, corr_thres);
    ptr_evaluator = std::make_unique<ContLCDEvaluator>(fpath_sens_gt_pose, fpath_lidar_bins, corr_thres);   //导入kitti数据，处理

    yl.loadOneConfig({"ContourDBConfig", "nnk_"}, db_config.nnk_);
    yl.loadOneConfig({"ContourDBConfig", "max_fine_opt_"}, db_config.max_fine_opt_);
    yl.loadSeqConfig({"ContourDBConfig", "q_levels_"}, db_config.q_levels_);

    yl.loadOneConfig({"ContourDBConfig", "TreeBucketConfig", "max_elapse_"}, db_config.tb_cfg_.max_elapse_);
    yl.loadOneConfig({"ContourDBConfig", "TreeBucketConfig", "min_elapse_"}, db_config.tb_cfg_.min_elapse_);

    yl.loadOneConfig({"ContourDBConfig", "ContourSimThresConfig", "ta_cell_cnt"}, db_config.cont_sim_cfg_.ta_cell_cnt);
    yl.loadOneConfig({"ContourDBConfig", "ContourSimThresConfig", "tp_cell_cnt"}, db_config.cont_sim_cfg_.tp_cell_cnt);
    yl.loadOneConfig({"ContourDBConfig", "ContourSimThresConfig", "tp_eigval"}, db_config.cont_sim_cfg_.tp_eigval);
    yl.loadOneConfig({"ContourDBConfig", "ContourSimThresConfig", "ta_h_bar"}, db_config.cont_sim_cfg_.ta_h_bar);
    yl.loadOneConfig({"ContourDBConfig", "ContourSimThresConfig", "ta_rcom"}, db_config.cont_sim_cfg_.ta_rcom);
    yl.loadOneConfig({"ContourDBConfig", "ContourSimThresConfig", "tp_rcom"}, db_config.cont_sim_cfg_.tp_rcom);
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

    yl.loadSeqConfig({"ContourManagerConfig", "lv_grads_"}, cm_config.lv_grads_);
    yl.loadOneConfig({"ContourManagerConfig", "reso_row_"}, cm_config.reso_row_);
    yl.loadOneConfig({"ContourManagerConfig", "reso_col_"}, cm_config.reso_col_);
    yl.loadOneConfig({"ContourManagerConfig", "n_row_"}, cm_config.n_row_);
    yl.loadOneConfig({"ContourManagerConfig", "n_col_"}, cm_config.n_col_);
    yl.loadOneConfig({"ContourManagerConfig", "lidar_height_"}, cm_config.lidar_height_);
    yl.loadOneConfig({"ContourManagerConfig", "blind_sq_"}, cm_config.blind_sq_);
    yl.loadOneConfig({"ContourManagerConfig", "min_cont_key_cnt_"}, cm_config.min_cont_key_cnt_);
    yl.loadOneConfig({"ContourManagerConfig", "min_cont_cell_cnt_"}, cm_config.min_cont_cell_cnt_);
    yl.loadOneConfig({"ContourManagerConfig", "piv_firsts_"}, cm_config.piv_firsts_);
    yl.loadOneConfig({"ContourManagerConfig", "dist_firsts_"}, cm_config.dist_firsts_);
    yl.loadOneConfig({"ContourManagerConfig", "roi_radius_"}, cm_config.roi_radius_);

    yl.loadOneConfig({"fpath_outcome_sav"}, sav_path);

    yl.close();
  }

  ///
  /// \param outer_cnt
  /// \return 0: normal. <0: external signal. 1: load failed、
  //这个函数包含了一次整个位置识别 匹配工作 是主要工作函数
  int spinOnce(int &outer_cnt) {
    // CHECK(ptr_contour_db && ptr_evaluator);
    mtx_status.lock();
    if (stat_terminated) {
      printf("Spin terminated by external signal.\n");
      mtx_status.unlock();
      return -1;
    }
    if (stat_paused) {
      printf("Spin paused by external signal.\n");
      mtx_status.unlock();
      return -2;
    }
    mtx_status.unlock();

    bool loaded = ptr_evaluator->loadNewScan();   //查询是否有新的点云帧，内部有全局计数变量，打印并返回查询状态
    if (!loaded) {
      printf("Load new scan failed.\n");
      return 1;
    }
    TicToc clk;
    ros::Time wall_time_ros = ros::Time::now();
    outer_cnt++;    //? 这个是什么作用

    // 1. Init current scan

    stp.lap();    //帧计数
    stp.start();  
    // std::shared_ptr<ContourManager> ptr_cm_tgt = ptr_evaluator->getCurrContourManager(cm_config);   //定义查询描述符指针，制作描述符,返回描述符指针
    std::shared_ptr<ContourManager> ptr_cm_tgt = nullptr;      //TODO 在这里替换上获取描述符的函数
    // stp.record("make bev");   //记录制作描述符的时间
    const auto laser_info_tgt = ptr_evaluator->getCurrScanInfo();
    // printf("\n===\nLoaded: assigned seq: %d, bin path: %s\n", laser_info_tgt.seq, laser_info_tgt.fpath.c_str());

    // 1.1 Prepare and display info: gt/shifted pose, tf
    double ts_curr = laser_info_tgt.ts;
    if (ts_beg < 0) ts_beg = ts_curr;

    Eigen::Isometry3d T_gt_curr = laser_info_tgt.sens_pose;
    Eigen::Vector3d time_translate(0, 0, 10);
    time_translate = time_translate * (ts_curr - ts_beg) / 60;  // 10m per min     这里分析是每1分钟偏移10m time_translate 不是z轴偏移
    g_poses.insert(std::make_pair(laser_info_tgt.seq, GlobalPoseInfo(T_gt_curr, time_translate.z())));

#if PUB_ROS_MSG
    geometry_msgs::TransformStamped tf_gt_curr = tf2::eigenToTransform(T_gt_curr);
    broadcastCurrPose(tf_gt_curr);  // the stamp is now

    tf_gt_curr.header.seq = laser_info_tgt.seq;
    tf_gt_curr.transform.translation.z += time_translate.z();
    publishPath(wall_time_ros, tf_gt_curr);
    if (laser_info_tgt.seq % 50 == 0)  // It is laggy to display too many characters in rviz
      publishScanSeqText(wall_time_ros, tf_gt_curr, laser_info_tgt.seq);
#endif

    // 1.2. save images of layers

#if SAVE_MID_FILE
    clk.tic();
    for (int i = 0; i < cm_config.lv_grads_.size(); i++) {
      std::string f_name = PROJ_DIR + "/results/layer_img/contour_" + "lv" + std::to_string(i) + "_" +
                           ptr_cm_tgt->getStrID() + ".png";   // TODO: what should be the str name of scans?
      ptr_cm_tgt->saveContourImage(f_name, i);
    }
    std::cout << "Time save layers: " << clk.toctic() << std::endl;
#endif
    ptr_cm_tgt->clearImage();  // a must to save memory

    // 2. query
    std::vector<std::pair<int, int>> new_lc_pairs;
    std::vector<bool> new_lc_tfp;
    std::vector<std::shared_ptr<const ContourManager>> ptr_cands;
    std::vector<double> cand_corr;
    std::vector<Eigen::Isometry2d> bev_tfs;

    clk.tic();
    int has_cand_flag = ptr_contour_db->queryRangedKNN(ptr_cm_tgt, thres_lb_, thres_ub_, ptr_cands, cand_corr, bev_tfs);  //查询检索获取候选
    // printf("%lu Candidates in %7.5fs: \n", ptr_cands.size(), clk.toc());

//    if(laser_info_tgt.seq == 894){
//      printf("Manual break point here.\n");
//    }

    //测试真值帧与匹配帧的关系
    static int cand_yes_lc_yes = 0;
    static int cand_yes = 0;
    if(has_cand_flag != 0)
    {
      cand_yes++;
      if((ptr_evaluator->getCurrScanInfo()).has_gt_positive_lc)
        cand_yes_lc_yes++;
    }
    std::cout << "\t" << "\t" << "cand nums: " << cand_yes << " cand & lc nums: " << cand_yes_lc_yes << std::endl;


    // 2.1 process query results
    CHECK(ptr_cands.size() < 2);
    PredictionOutcome pred_res;
    if (ptr_cands.empty())
      pred_res = ptr_evaluator->addPrediction(ptr_cm_tgt, 0.0);
    else {
      pred_res = ptr_evaluator->addPrediction(ptr_cm_tgt, cand_corr[0], ptr_cands[0], bev_tfs[0]);  //求解查询帧的prediction outcome
      if (pred_res.tfpn == PredictionOutcome::TP || pred_res.tfpn == PredictionOutcome::FP) {       //储存检索回环的帧id
        new_lc_pairs.emplace_back(ptr_cm_tgt->getIntID(), ptr_cands[0]->getIntID());
        new_lc_tfp.emplace_back(pred_res.tfpn == PredictionOutcome::TP);
#if SAVE_MID_FILE
        // save images of pairs
        std::string f_name =
            PROJ_DIR + "/results/match_comp_img/lc_" + ptr_cm_tgt->getStrID() + "-" + ptr_cands[0]->getStrID() +
            ".png";
        ContourManager::saveMatchedPairImg(f_name, *ptr_cm_tgt, *ptr_cands[0]);   //保存文件
        printf("Image saved: %s-%s\n", ptr_cm_tgt->getStrID().c_str(), ptr_cands[0]->getStrID().c_str());
#endif
      }
    }


    //打印tp fp tn fn
    switch (pred_res.tfpn) {
      case PredictionOutcome::TP:
        printf("Prediction outcome: TP\n");
        cnt_tp++;
        break;
      case PredictionOutcome::FP:
        printf("Prediction outcome: FP\n");
        cnt_fp++;
        break;
      case PredictionOutcome::TN:
        printf("Prediction outcome: TN\n");
        break;
      case PredictionOutcome::FN:
        printf("Prediction outcome: FN\n");
        cnt_fn++;
        break;
    }

    //这里打印各个tp帧的error平均值
    printf("TP Error mean: t:%7.4f m, r:%7.4f rad\n", ptr_evaluator->getTPMeanTrans(), ptr_evaluator->getTPMeanRot());
    printf("TP Error rmse: t:%7.4f m, r:%7.4f rad\n", ptr_evaluator->getTPRMSETrans(), ptr_evaluator->getTPRMSERot());
    printf("Accumulated tp poses: %d\n", cnt_tp);
    printf("Accumulated fn poses: %d\n", cnt_fn);
    printf("Accumulated fp poses: %d\n", cnt_fp);

    stp.start();
    // 3. update database
    // add scan
    ptr_contour_db->addScan(ptr_cm_tgt, laser_info_tgt.ts);     //保存描述符数据 这里没有隔断相邻的点云帧吗？
    // balance
    clk.tic();
    ptr_contour_db->pushAndBalance(laser_info_tgt.seq, laser_info_tgt.ts);    //放入检索树
    stp.record("Update database");
    printf("Rebalance tree cost: %7.5f\n", clk.toc());

#if PUB_ROS_MSG
    // 4. publish new vis
    publishLCConnections(new_lc_pairs, new_lc_tfp, wall_time_ros);
#endif

    return 0;
  }

  //数据保存函数，将处理得到的数据对应保存到各个文件
  void savePredictionResults(const std::string &sav_path) const {
    ptr_evaluator->savePredictionResults(sav_path);
  }

  inline int get_tp() const { return cnt_tp; }

  inline int get_fp() const { return cnt_fp; }

  inline int get_fn() const { return cnt_fn; }
};

pcl::PointXYZRGB vec2point(const Eigen::Vector3d &vec, std::vector<std::uint8_t> &rgb) {
  pcl::PointXYZRGB pi;
  pi.x = vec[0];
  pi.y = vec[1];
  pi.z = vec[2];
  pi.r = rgb[0];  // 红色
  pi.g = rgb[1];    // 绿色
  pi.b = rgb[2];    // 蓝色
  return pi;
}
Eigen::Vector3d point2vec(const pcl::PointXYZRGB &pi) {
  return Eigen::Vector3d(pi.x, pi.y, pi.z);
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr readCloud(std::string lidar_path, Eigen::Isometry3d pose, std::vector<std::uint8_t> rgb)
{
  // Read KITTI data
  std::ifstream lidar_data_file;
  std::vector<float> lidar_data_buffer = {};
  lidar_data_file.open(lidar_path,
                       std::ifstream::in | std::ifstream::binary);
  if (!lidar_data_file) {
    std::cout << "Read End..." << std::endl;
    return nullptr;
  }
  else{
    lidar_data_file.seekg(0, std::ios::end);
    const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
    lidar_data_file.seekg(0, std::ios::beg);

    lidar_data_buffer.resize(num_elements);
    lidar_data_file.read(reinterpret_cast<char *>(&lidar_data_buffer[0]),
                         num_elements * sizeof(float));
  }
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>());
  for (std::size_t i = 0; i < lidar_data_buffer.size(); i += 4) {
    pcl::PointXYZRGB point;
    point.x = lidar_data_buffer[i];
    point.y = lidar_data_buffer[i + 1];
    point.z = lidar_data_buffer[i + 2];
    Eigen::Vector3d pv = point2vec(point);
    pv = pose.rotation() * pv + pose.translation();
    point = vec2point(pv,rgb);
    temp_cloud->push_back(point);
  }
  return temp_cloud;
}


int main(int argc, char **argv) {

//TODO mqtt部分未整理
    try {
        // 创建 MQTT 客户端并连接
        auto client = connectToBroker();

        // 调用接收函数获取数据
        std::vector<LECD> received_data = receiveMqttData(client);

        // 打印接收到的结构体数据
        std::cout << "Received LECD data:" << std::endl;
        for (const auto& lecd : received_data) {
            std::cout << "Name: " << lecd.name << ", Age: " << lecd.age << ", Skills: ";
            for (const auto& skill : lecd.skills) {
                std::cout << skill << " ";
            }
            std::cout << std::endl;
        }

    } catch (const std::exception& exc) {
        std::cerr << "Program error: " << exc.what() << std::endl;
        return 1;
    }

  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);

  ros::init(argc, argv, "batch_bin_test");
  ros::NodeHandle nh;

  printf("batch bin test start\n");

  ros::Publisher pubOdomAftMapped =
    nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);    //输出odom
  ros::Publisher pubCureentCloud =                                    //点云
    nh.advertise<sensor_msgs::PointCloud2>("/cloud_current", 100);
  ros::Publisher pubMatchCloud =                                    //tp点云
    nh.advertise<sensor_msgs::PointCloud2>("/cloud_key_points", 100);
  ros::Publisher pubPath = 
    nh.advertise<nav_msgs::Path>("/lecd_path", 10);
  ros::Publisher pubtpfp =
      nh.advertise<visualization_msgs::MarkerArray>("descriptor_line", 10);

  nav_msgs::Path globalPath;
  visualization_msgs::MarkerArray marker_array;
  int marker_id = 0;
  std::vector<Eigen::Isometry3d> his_pose_display;    //储存演示用的位置
  int frame_cnt = -1;
  
  // Check thres path
//  std::string cand_score_config = PROJ_DIR + "/config/score_thres_kitti_bag_play.cfg";
  std::string cand_score_config = PROJ_DIR + "/config/batch_bin_test_config.yaml";

  // Main process:
  BatchBinSpinner o(nh);                                //这个是主要的数据结构 初始化仅传递一个ros句柄

  std::string fpath_outcome_sav;
  o.loadConfig(cand_score_config, fpath_outcome_sav);   //加载参数，直接赋值到对应的数据结构中去

  stp = SequentialTimeProfiler(fpath_outcome_sav);    //初始化顺序时间档案器

  ros::Rate rate(300);
  int cnt = 0;


  printf("\nHold for 3 seconds...\n");
  std::this_thread::sleep_for(std::chrono::duration<double>(3.0));  // human readability: have time to see init output

  while (ros::ok()) {
    ros::spinOnce();

    int ret_code = o.spinOnce(cnt);   //这里是主要的实现函数 返回值=0
    
    if (ret_code == -2 || ret_code == 1)
      ros::Duration(1.0).sleep();   //延时1s
    else if (ret_code == -1)
      break;

    rate.sleep();
  }

  // plt::show();


  o.savePredictionResults(fpath_outcome_sav);   //数据保存

  stp.printScreen(true);      //打印最后的耗时统计
  const std::string log_dir = PROJ_DIR + "/log/";
  stp.printFile(log_dir + "timing_cont2.txt", true);


  return 0;
}