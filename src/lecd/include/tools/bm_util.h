//
// Created by lewis on 5/5/22.
//

#ifndef CONT2_BM_UTIL_H
#define CONT2_BM_UTIL_H

//
// Created by lewis on 10/21/21.
// Benchmark utility
//
#include <chrono>
#include <algorithm>
#include <string>
#include <numeric>
#include <map>
#include <utility>
#include <sys/time.h>
#include <iostream>
#include <fstream>



class TicToc {
public:
  TicToc() {
    tic();
  }

  void tic() {
    start = std::chrono::steady_clock::now();
  }

  double toc() {
    end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    return elapsed_seconds.count();
  }

  double toctic() {
    double ret = toc();
    tic();
    return ret;
  }

private:
  std::chrono::time_point<std::chrono::steady_clock> start, end;
};


extern std::unordered_map<int, std::array<long, 4>> cp_stamps;    //压缩部分时间戳
extern std::mutex stamps_mtx;  // 互斥锁
//这个似乎是完成检测后的评估整理打印class
class SequentialTimeProfiler {
protected:
  struct OneLog {   //TODO 待分析各个变量的含义
    int idx{}, cnt{};
    double samps{};
    double autocorrs{};
    std::vector<double> all_samps;


    OneLog() = default;

    explicit OneLog(int i, int a, double b) : idx(i), cnt(a), samps(b), autocorrs(b * b) {all_samps.emplace_back(b);}
    
    explicit OneLog(int i, int a, double b, std::vector<double> al_samps) : idx(i), cnt(a), samps(b), autocorrs(b * b) 
    {all_samps.insert(all_samps.end(), al_samps.begin(), al_samps.end());}
  };

  TicToc clk;
  std::map<std::string, OneLog> logs;
  int cnt_loops = 0;
  size_t max_len = 5;   // min name length
  std::string desc = "";  // short description

public:

  SequentialTimeProfiler() = default;

  SequentialTimeProfiler(const std::string &name) : desc(name) {};

  inline std::string getDesc() const {
    return desc;
  }

  static std::string getTimeString() {
    std::time_t now = std::time(nullptr);
    struct tm tstruct = *std::localtime(&now);
    char buf[80];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %a %X %z", &tstruct);
    return buf;
  }

  inline void start() {
    clk.tic();
  }

  /// record and reset timer
  /// \param name The name of a log entry
  //记录各个步骤的时间并保存，传入各个步骤的名字
  void record(const std::string &name) {
    const double dt = clk.toc();
    auto it = logs.find(name);
    if (it == logs.end()) {   //log里面不存在该步骤名称 则新建一个
      logs[name] = OneLog(static_cast<int>(logs.size()), 1, dt);
      max_len = std::max(max_len, name.length());   //更新最大步骤名称长度
    } else {
      it->second.cnt++;
      it->second.samps += dt;   //求当前步骤在每次循环的时间和
      it->second.autocorrs += dt * dt;
      it->second.all_samps.emplace_back(dt);
    }
    clk.tic();  // auto reset, useful for sequential timing.
  }

  /// record and reset timer
  /// \param name The name of a log entry
  // //记录各个步骤的时间并保存，传入各个步骤的名字 并返回步骤耗时
  // void record(const std::string &name, double &dt_curr) {
  //   const double dt = clk.toc();
  //   auto it = logs.find(name);
  //   if (it == logs.end()) {
  //     logs[name] = OneLog(static_cast<int>(logs.size()), 1, dt);
  //     max_len = std::max(max_len, name.length());
  //   } else {
  //     it->second.cnt++;
  //     it->second.samps += dt;
  //     it->second.autocorrs += dt * dt;
  //   }
  //   dt_curr = dt;
  //   clk.tic();  // auto reset, useful for sequential timing.
  // }

  //记录各个步骤的时间并保存，传入各个步骤的名字和耗时
  void record(const std::string &name, double &dt_dur) {
    // const double dt = clk.toc();
    auto it = logs.find(name);
    if (it == logs.end()) {   //log里面不存在该步骤名称 则新建一个
      logs[name] = OneLog(static_cast<int>(logs.size()), 1, dt_dur);
      max_len = std::max(max_len, name.length());   //更新最大步骤名称长度
    } else {
      it->second.cnt++;
      it->second.samps += dt_dur;   //求当前步骤在每次循环的时间和
      it->second.autocorrs += dt_dur * dt_dur;
      it->second.all_samps.emplace_back(dt_dur);
    }
    clk.tic();  // auto reset, useful for sequential timing.
  }

  inline void lap() {
    cnt_loops++;
  }

  //总结数据打印
  void printScreen(bool sort_by_cost = false) const {
    printf("\n=== Time Profiling @%s ===\n", getTimeString().c_str());
    printf("=== Description: %s\n", desc.c_str());
    printf("%5s %s %10s %10s %10s %10s %10s %10s\n",
           "Index", (std::string(max_len - 4, ' ') + "Name").c_str(),
           "Count", "Average", "Stddev", "Per loop", "Loop %", "Accum %");
    std::vector<std::pair<std::string, OneLog>> vec(logs.begin(), logs.end());
    if (sort_by_cost)
      std::sort(vec.begin(), vec.end(),
                [&](const std::pair<std::string, OneLog> &e1, const std::pair<std::string, OneLog> &e2) {
                  return e1.second.samps > e2.second.samps;
                });
    else
      std::sort(vec.begin(), vec.end(),
                [&](const std::pair<std::string, OneLog> &e1, const std::pair<std::string, OneLog> &e2) {
                  return e1.second.idx < e2.second.idx;
                });

    double t_total = 0, t_accum = 0;
    for (const auto &itm: vec)
      t_total += itm.second.samps;

    for (const auto &itm: vec) {
      const auto &lg = itm.second;
      double x_bar = lg.samps / lg.cnt;
      double stddev = 0;
      if (lg.cnt > 1)
        stddev = std::sqrt(1.0 / (lg.cnt - 1) * (lg.autocorrs + lg.cnt * x_bar * x_bar - 2 * x_bar * lg.samps));
      t_accum += lg.samps;
      printf("%5d %s %10d %10.2e %10.2e %10.2e %10.2f %10.2f\n",
             lg.idx,
             (std::string(max_len - itm.first.length(), ' ') + itm.first).c_str(),
             lg.cnt,
             x_bar,
             stddev,
             cnt_loops > 0 ? lg.samps / cnt_loops : 0,
             lg.samps / t_total * 100, // count_i * avg_i / (\sum(count*avg))
             t_accum / t_total * 100
      );
    }
    printf("%5s %s %10d %10s %10s %10.2e %10s %10s\n",
           "*", (std::string(max_len - 4, ' ') + "*sum").c_str(), cnt_loops, "*", "*",
           cnt_loops > 0 ? t_total / cnt_loops : 0,
           "*", "*"
    );
  }

  void printFile(const std::string &fpath, bool sort_by_cost = false) const {
    std::FILE *fp;
    fp = std::fopen(fpath.c_str(), "a");

    fprintf(fp, "\n=== Time Profiling @%s ===\n", getTimeString().c_str());
    fprintf(fp, "=== Description: %s\n", desc.c_str());
    fprintf(fp, "%5s %s %10s %10s %10s %10s %10s %10s\n",
            "Index", (std::string(max_len - 4, ' ') + "Name").c_str(),
            "Count", "Average", "Stddev", "Per loop", "Loop %", "Accum %");
    std::vector<std::pair<std::string, OneLog>> vec(logs.begin(), logs.end());
    if (sort_by_cost)
      std::sort(vec.begin(), vec.end(),
                [&](const std::pair<std::string, OneLog> &e1, const std::pair<std::string, OneLog> &e2) {
                  return e1.second.samps > e2.second.samps;
                });
    else
      std::sort(vec.begin(), vec.end(),
                [&](const std::pair<std::string, OneLog> &e1, const std::pair<std::string, OneLog> &e2) {
                  return e1.second.idx < e2.second.idx;
                });

    double t_total = 0, t_accum = 0;
    for (const auto &itm: vec)
      t_total += itm.second.samps;

    for (const auto &itm: vec) {
      const auto &lg = itm.second;
      double x_bar = lg.samps / lg.cnt;
      double stddev = 0;
      if (lg.cnt > 1)
        stddev = std::sqrt(1.0 / (lg.cnt - 1) * (lg.autocorrs + lg.cnt * x_bar * x_bar - 2 * x_bar * lg.samps));
      t_accum += lg.samps;
      fprintf(fp, "%5d %s %10d %10.2e %10.2e %10.2e %10.2f %10.2f\n",
              lg.idx,
              (std::string(max_len - itm.first.length(), ' ') + itm.first).c_str(),
              lg.cnt,
              x_bar,
              stddev,
              cnt_loops > 0 ? lg.samps / cnt_loops : 0,
              lg.samps / t_total * 100, // count_i * avg_i / (\sum(count*avg))
              t_accum / t_total * 100
      );
    }
    fprintf(fp, "%5s %s %10d %10s %10s %10.2e %10s %10s\n",
            "*", (std::string(max_len - 4, ' ') + "*sum").c_str(), cnt_loops, "*", "*",
            cnt_loops > 0 ? t_total / cnt_loops : 0,
            "*", "*"
    );
    std::fclose(fp);
  }

  void SaveConsumingTime(const std::string &fpath) const
  {
    std::vector<std::pair<std::string, OneLog>> vec_consum(logs.begin(), logs.end());
    
    std::fstream res_file(fpath, std::ios::out);
    if (res_file.rdstate() != std::ifstream::goodbit)
    {
        std::cerr << "Error opening " << fpath << std::endl;
        return;
    }
    std::sort(vec_consum.begin(), vec_consum.end(),
              [&](const std::pair<std::string, OneLog> &e1, const std::pair<std::string, OneLog> &e2) {
                return e1.second.all_samps.size() > e2.second.all_samps.size();
              });
    int nums = vec_consum[0].second.all_samps.size();
    std::cout << "logs size: " << vec_consum.size() << " all samps size: " << nums << std::endl;

    for(const auto &log_ : vec_consum)
    {
      // std::cout << log_.first << "\t";

      res_file << log_.first << "\t";
    }
    res_file << "\n";
    // std::cout << std::endl;

    for(int i = 0; i < nums; i++)
    {
      for(const auto &log_ : vec_consum)
      {
        res_file << log_.second.all_samps[i] << "\t";
      }      
      res_file << "\n";
    }

    res_file.close();
  }


  //ntp时间戳部分，用于接收队列前储存ntp时间戳
  long getCurrentStamp(){
      struct timeval tv;
      long result;
      gettimeofday(&tv, NULL); // 获取当前时间
      result = tv.tv_sec * 1000 * 1000 + tv.tv_usec;       
      return result; 
  }
    //传入 点云序号 时间戳，时间戳序号
  void pushstamps(int pt_id, long stamp_1, long stamp_2, long stamp_3)
  {
    std::lock_guard<std::mutex> lock(stamps_mtx); 
    // 通过 id 检索数据
    auto found = cp_stamps.find(pt_id);
    if (found == cp_stamps.end()) {
      std::array<long, 4> arr = {};
      arr[0] = stamp_1;
      arr[1] = stamp_2;
      arr[2] = stamp_3;
      arr[3] = getCurrentStamp();
      cp_stamps[pt_id] = arr;
    }     
  }

// 合并stp部分，用于多线程的stp合并

  // //lap合并添加
  // void addlap(int &add)
  // {
  //   cnt_loops += add;
  // }

  // int getlap()
  // {
  //   return cnt_loops;
  // }

  //OneLog合并
  // 合并两个 SequentialTimeProfiler 的日志
  void addLogs(const SequentialTimeProfiler &other) {
      //添加lap
      cnt_loops += other.cnt_loops;
      // 遍历 other 的 logs，并将其合并到当前对象的 logs 中
      for (const auto &entry : other.logs) {
          const std::string &name = entry.first;
          const OneLog &log = entry.second;

          auto it = logs.find(name);
          if (it == logs.end()) {
              // 如果当前 logs 中没有该名称，则直接插入
              logs[name] = OneLog(static_cast<int>(logs.size()), log.cnt, log.samps, log.all_samps);
              max_len = std::max(max_len, name.length());
              auto it_again = logs.find(name);
              it_again->second.autocorrs += log.autocorrs;
          } else {
              // 如果已经存在该名称，则合并
              it->second.cnt += log.cnt;
              it->second.samps += log.samps;
              it->second.autocorrs += log.autocorrs;
              it->second.all_samps.insert(it->second.all_samps.end(), log.all_samps.begin(), log.all_samps.end());
          }
      }
  }


  void compressrecord(int pt_id);
  // {
  //   auto found = cp_stamps.find(pt_id);
  //   if (found != cp_stamps.end()) {    
  //     double dur_;
  //     dur_ = (found->second[1] - found->second[0])  > 0 ? (found->second[1] - found->second[0]) / 1000000.0 : 0;
  //     record("compress",dur_);

  //     // dur_ = (found->second[2] - found->second[1])  > 0 ? (found->second[2] - found->second[1]) / 1000000.0 : 0;
  //     // record("cp queue",dur_);

  //     dur_ = (found->second[3] - found->second[1])  > 0 ? (found->second[3] - found->second[1]) / 1000000.0 : 0;
  //     record("transmit",dur_);
  //   }
  // }
};

#endif //CONT2_BM_UTIL_H
