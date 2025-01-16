#ifndef LIDAR_REC_H
#define LIDAR_REC_H

#include <mqtt/client.h>
#include "contour.h"
#include <iostream>
#include <string>
#include <chrono>
#include <memory>
#include <nlohmann/json.hpp>
#include <vector>
#include <mutex>
#include <thread>
#include <condition_variable>

// 定义 MQTT Broker 地址和话题
const std::string SERVER_ADDRESS{"tcp://8.138.105.82:1883"};
const std::string CLIENT_ID{"LECD_rec_subscriber"};
const std::string TOPIC{"lecd_topic"};
const int QOS = 1;  // QoS级别
const auto TIMEOUT = std::chrono::seconds(1);


// 为 Eigen 矩阵类型添加 JSON 支持
namespace nlohmann {
    template <typename Scalar, int Rows, int Cols>
    struct adl_serializer<Eigen::Matrix<Scalar, Rows, Cols>> {
        // 序列化 Eigen 矩阵 -> JSON 数组
        static void to_json(nlohmann::json& j, const Eigen::Matrix<Scalar, Rows, Cols>& matrix) {
            std::vector<std::vector<Scalar>> mat_vec(matrix.rows(), std::vector<Scalar>(matrix.cols()));
            for (int i = 0; i < matrix.rows(); ++i) {
                for (int j = 0; j < matrix.cols(); ++j) {
                    mat_vec[i][j] = matrix(i, j);
                }
            }
            j = mat_vec;  // 转换为 JSON 数组
        }

        // 反序列化 JSON 数组 -> Eigen 矩阵
        static void from_json(const nlohmann::json& j, Eigen::Matrix<Scalar, Rows, Cols>& matrix) {
            std::vector<std::vector<Scalar>> mat_vec = j.get<std::vector<std::vector<Scalar>>>();
            if (!mat_vec.empty()) {
                matrix.resize(mat_vec.size(), mat_vec[0].size());
                for (int i = 0; i < matrix.rows(); ++i) {
                    for (int j = 0; j < matrix.cols(); ++j) {
                        matrix(i, j) = mat_vec[i][j];
                    }
                }
            }
        }
    };
}
struct ECD {

    int16_t level_;   //lay level 
    int16_t poi_[2]; // a point in full bev coordinate belonging to this contour/slice. 在bev下的像素坐标？

    // statistical summary
    int16_t cell_cnt_{};    //椭圆占据的网格数量
    V2F pos_mean_;          //xy均值 中心点
    V2F eig_vals_;          //协方差特征值
    M2F eig_vecs_; // gaussian ellipsoid axes. if ecc_feat_==false, this is meaningless  特征值对应的椭球轴(特征向量) 长轴匹配+生成协方差 ecc_feat_无效则为无意义值
    float eccen_{};   // 0: circle    //偏心率，0为圆形
    float vol3_mean_{};     //高度均值 用于统计数据计算中
    bool ecc_feat_ = false;   // eccentricity large enough (with enough cell count)   椭圆有效（足够大，离心率足够大）判断    


    // 为 ECD 定义序列化函数
    friend void to_json(nlohmann::json& j, const ECD& obj) {
        j = nlohmann::json{
            {"level_", obj.level_},
            {"poi_", {obj.poi_[0], obj.poi_[1]}},
            {"cell_cnt_", obj.cell_cnt_},
            {"pos_mean_", obj.pos_mean_},
            {"eig_vals_", obj.eig_vals_},
            {"eig_vecs_", obj.eig_vecs_},
            {"eccen_", obj.eccen_},
            {"vol3_mean_", obj.vol3_mean_},
            {"ecc_feat_", obj.ecc_feat_}
        };
    }

    // 为 ECD 定义反序列化函数
    friend void from_json(const nlohmann::json& j, ECD& obj) {
        j.at("level_").get_to(obj.level_);
        j.at("poi_").get_to(obj.poi_);
        j.at("cell_cnt_").get_to(obj.cell_cnt_);

        // 处理 pos_mean_ (V2F 类型)
        if (j.contains("pos_mean_")) {
            std::vector<std::vector<float>> pos_mean_vec = j.at("pos_mean_").get<std::vector<std::vector<float>>>();
            if (pos_mean_vec.size() == 2 && pos_mean_vec[0].size() == 1 && pos_mean_vec[1].size() == 1) {
                obj.pos_mean_(0) = pos_mean_vec[0][0]; // 取出第一个值
                obj.pos_mean_(1) = pos_mean_vec[1][0]; // 取出第二个值
            } else {
                throw std::runtime_error("Invalid pos_mean_ format");
            }
        }

        // 处理 eig_vals_ (V2F 类型)
        if (j.contains("eig_vals_")) {
            std::vector<std::vector<float>> eig_vals_vec = j.at("eig_vals_").get<std::vector<std::vector<float>>>();
            if (eig_vals_vec.size() == 2 && eig_vals_vec[0].size() == 1 && eig_vals_vec[1].size() == 1) {
                obj.eig_vals_(0) = eig_vals_vec[0][0]; // 取出第一个值
                obj.eig_vals_(1) = eig_vals_vec[1][0]; // 取出第二个值
            } else {
                throw std::runtime_error("Invalid eig_vals_ format");
            }
        }


        // 处理 eig_vecs_ (M2F 类型)
        if (j.contains("eig_vecs_")) {
            std::vector<std::vector<float>> eig_vecs_vec = j.at("eig_vecs_").get<std::vector<std::vector<float>>>();
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                    obj.eig_vecs_(i, j) = eig_vecs_vec[i][j];
                }
            }
        }

        j.at("eccen_").get_to(obj.eccen_);
        j.at("vol3_mean_").get_to(obj.vol3_mean_);
        j.at("ecc_feat_").get_to(obj.ecc_feat_);
    }
};

// 定义 LECD 结构体
struct LECD{
    std::vector<ECD> ellipse_gp;
    std::vector<std::vector<std::array<float, 10>>> keys_;   //检索键值
    //辅助数据 点云序号，椭圆数量等
    int16_t pt_seq;     //点云id
    int16_t lecd_nums;  //椭圆数量
    double time_stamp;  //时间戳

    friend void to_json(nlohmann::json& j, const LECD& obj) {
        j = nlohmann::json{
            {"pt_seq",obj.pt_seq},
            {"lecd_nums",obj.lecd_nums},
            {"time_stamp",obj.time_stamp},
            {"ellipse_gp", obj.ellipse_gp},  // 序列化 ellipse_gp
            {"keys_", nlohmann::json::array()}    // 初始化 keys_ 的 JSON 数组
        };
    
        // 序列化 keys_
        for (const auto& outer : obj.keys_) {
            nlohmann::json outer_array;
            for (const auto& inner : outer) {
                outer_array.push_back(inner);  // inner 是 std::array<float, 10>
            }
            j["keys_"].push_back(outer_array);
        }

    }

    friend void from_json(const nlohmann::json& j, LECD& obj) {
        // 解析 ellipse_gp
        j.at("pt_seq").get_to(obj.pt_seq);
        j.at("lecd_nums").get_to(obj.lecd_nums);
        j.at("time_stamp").get_to(obj.time_stamp);
        j.at("ellipse_gp").get_to(obj.ellipse_gp);

        // 解析 keys_
        obj.keys_.clear();
        for (const auto& outer_array : j.at("keys_")) {
            std::vector<std::array<float, 10>> outer_vector;
            for (const auto& inner_array : outer_array) {
                std::array<float, 10> inner{};
                for (size_t i = 0; i < inner.size(); ++i) {
                    inner[i] = inner_array.at(i).get<float>();
                }
                outer_vector.push_back(inner);
            }
            obj.keys_.push_back(outer_vector);
        }
    }    

};


//线程安全队列
class ThreadSafeQueue {
private:
    std::queue<LECD> queue_;
    std::mutex mutex_;
    std::condition_variable cond_var_;

public:
    void push(const LECD& data) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(data);
        cond_var_.notify_one(); // 通知等待的线程
    }

    LECD pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_var_.wait(lock, [this]() { return !queue_.empty(); }); // 等待队列非空
        LECD data = queue_.front();
        queue_.pop();
        return data;
    }

    bool is_empty() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
};

// 回调类，用于处理接收到的消息
class my_callback : public mqtt::callback {
    mqtt::async_client& client_;
    mqtt::topic& sub_topic_;
    ThreadSafeQueue& queue_; // 引用线程安全队列

public:
    explicit my_callback(mqtt::async_client& client, mqtt::topic& sub_topic, ThreadSafeQueue& queue)
        : client_(client), sub_topic_(sub_topic), queue_(queue) {}

    void message_arrived(mqtt::const_message_ptr msg) override {
        try {
            // 解析消息的 payload
            std::string payload = msg->get_payload();
            // 打印接收到的消息内容
            // std::cout << "Raw payload: " << payload << std::endl;
            nlohmann::json json_data = nlohmann::json::parse(payload);
            LECD received_data = json_data.get<LECD>();
            // queue_.push(received_data); // 将数据推入队列
            // 打印接收到的数据
            std::cout << "Message received on topic: " << msg->get_topic() << std::endl;
            // std::cout << "Payload: " << payload << std::endl;

            //存入队列
            queue_.push(received_data);

        } catch (const std::exception& e) {
            std::cerr << "Error parsing or processing message: " << e.what() << std::endl;
        }
    }

    void connected(const std::string& cause) override {
        std::cout << "Connected to MQTT broker." << std::endl;
        sub_topic_.subscribe();
    }

    void connection_lost(const std::string& cause) override {
        std::cerr << "Connection lost: " << cause << std::endl;
    }
};


// 全局队列实例
void mqtt_receiver_thread();
extern ThreadSafeQueue lecd_queue;

#endif