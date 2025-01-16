#include "lidar_rec.h"


ThreadSafeQueue lecd_queue;

//mqtt接收线程函数
void mqtt_receiver_thread() {
    const std::string SERVER_ADDRESS = "tcp://8.138.105.82:1883"; // MQTT 服务器地址
    const std::string CLIENT_ID = "LECD_rec_subscriber";
    const std::string TOPIC = "lecd_topic";
    const int QOS = 1;

    mqtt::async_client client(SERVER_ADDRESS, CLIENT_ID);
    mqtt::connect_options connOpts;
    connOpts.set_clean_session(true);
    connOpts.set_user_name("LECD_rec_subscriber");
    connOpts.set_password("123456");

    mqtt::topic sub_topic(client, TOPIC, QOS);
    my_callback cb(client, sub_topic, lecd_queue);
    client.set_callback(cb);

    try {
        client.connect(connOpts)->wait();
        std::cout << "MQTT receiver thread running..." << std::endl;

        // 持续接收数据
        std::cout << "Waiting for messages... Press Ctrl+C to exit." << std::endl;
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    } catch (const mqtt::exception& exc) {
        std::cerr << "Error: " << exc.what() << std::endl;
    }
}