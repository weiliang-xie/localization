#ifndef LIDAR_COMPRESS_MQTT_H
#define LIDAR_COMPRESS_MQTT_H

#include <mqtt/client.h>
#include <iostream>
#include <string>
#include <chrono>
#include <memory>
#include <nlohmann/json.hpp>
#include <vector>

// 定义 LECD 结构体
struct LECD {
    std::string name;
    int age;
    std::vector<std::string> skills;

    // 自动 JSON 序列化支持
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(LECD, name, age, skills)
};

// 定义 MQTT Broker 地址和话题
const std::string SERVER_ADDRESS{"tcp://localhost:1883"};
const std::string CLIENT_ID{"mqtt_cpp_complex_subscriber"};
const std::string TOPIC{"complex_vector_topic"};
const int QOS = 1;  // QoS级别
const auto TIMEOUT = std::chrono::seconds(1);

// 定义接收函数
std::vector<LECD> receiveMqttData(std::shared_ptr<mqtt::async_client> client) {
    std::vector<LECD> received_data;

    // 创建一个回调类用于处理接收到的消息
    class my_callback : public mqtt::callback {
    public:
        my_callback(std::vector<LECD>& data) : data_(data) {}

        // 连接成功时的回调
        void connected(const std::string& cause) override {
            std::cout << "Connected to MQTT Broker!" << std::endl;
            // 订阅话题
            client_->subscribe(TOPIC, QOS);
        }

        // 连接丢失时的回调
        void connection_lost(const std::string& cause) override {
            std::cout << "Connection lost: " << cause << std::endl;
        }

        // 消息到达时的回调
        void message_arrived(mqtt::const_message_ptr msg) override {
            std::cout << "Received message on topic: " << msg->get_topic() << std::endl;

            // 打印原始消息内容
            std::string payload = msg->get_payload();
            std::cout << "Raw payload: " << payload << std::endl;

            // 处理接收到的消息（假设消息是一个 JSON 字符串）
            try {
                // 将消息的负载内容反序列化为 JSON
                nlohmann::json json_data = nlohmann::json::parse(payload);

                // 将反序列化的 JSON 数据转换为 LECD 类型
                std::vector<LECD> data = json_data.get<std::vector<LECD>>();

                // 将数据保存到外部容器
                data_.insert(data_.end(), data.begin(), data.end());
            } catch (const std::exception& exc) {
                std::cerr << "Error parsing message: " << exc.what() << std::endl;
            }
        }

        // 设置客户端
        void set_client(std::shared_ptr<mqtt::async_client> client) {
            client_ = client;
        }

    private:
        std::shared_ptr<mqtt::async_client> client_;  // 用于订阅和接收消息的客户端
        std::vector<LECD>& data_;  // 引用外部数据容器
    };

    try {
        // 设置回调
        my_callback callback(received_data);
        callback.set_client(client);

        // 设置回调
        client->set_callback(callback);

        // 连接到 MQTT Broker
        mqtt::connect_options connOpts;
        connOpts.set_clean_session(true);
        client->connect(connOpts)->wait();

        // 保持接收消息
        std::cout << "Waiting for messages... Press Ctrl+C to exit." << std::endl;
        // 暂停，等待接收到消息
        std::this_thread::sleep_for(std::chrono::seconds(10));  // 适当等待时间，保证能接收到消息

        // 断开连接
        client->disconnect()->wait();

    } catch (const std::exception& exc) {
        std::cerr << "Program error: " << exc.what() << std::endl;
    }

    return received_data;
}

// 连接到 MQTT Broker
std::shared_ptr<mqtt::async_client> connectToBroker() {
    auto client = std::make_shared<mqtt::async_client>(SERVER_ADDRESS, CLIENT_ID);

    mqtt::connect_options connOpts;
    connOpts.set_clean_session(true);

    try {
        std::cout << "Connecting to MQTT Broker..." << std::endl;
        client->connect(connOpts)->wait();
        std::cout << "Connected successfully!" << std::endl;
    } catch (const mqtt::exception& exc) {
        std::cerr << "Error connecting to broker: " << exc.what() << std::endl;
        throw;
    }

    return client;
}

#endif