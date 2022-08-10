#include <iostream>
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

typedef websocketpp::server<websocketpp::config::asio> server;

using websocketpp::lib::bind;
using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;

typedef server::message_ptr message_ptr;

void on_message(server *s, websocketpp::connection_hdl hdl, message_ptr msg)
{
    std::cout << "on_message called with hdl: " << hdl.lock().get()
              << " and message: " << msg->get_payload()
              << std::endl;

    if (msg->get_payload() == "stop-listening")
    {
        s->stop_listening();
        return;
    }

    try
    {
        s->send(hdl, msg->get_payload(), msg->get_opcode());
    }
    catch (websocketpp::exception const &e)
    {
        std::cout << "Echo failed because: "
                  << "(" << e.what() << ")" << std::endl;
    }
}

void *serverThread(void *params)
{
    server *myServer = (server *)params;

    myServer->run();

    return NULL;
}

int main()
{
    server myServer;

    try
    {
        pthread_attr_t attr;
        pthread_t thread;

        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

        myServer.set_access_channels(websocketpp::log::alevel::all);
        myServer.clear_access_channels(websocketpp::log::alevel::frame_payload);

        myServer.init_asio();

        myServer.set_message_handler(bind(&on_message, &myServer, ::_1, ::_2));

        myServer.listen(9002);
        myServer.start_accept();

        int rc = pthread_create(&thread, &attr, &serverThread, (void *)&myServer);
        if (rc)
        {
            printf("[ERROR] Couln't initialize WebSocket server\n");
            exit(-1);
        }
    }
    catch (websocketpp::exception const &e)
    {
        std::cout << e.what() << std::endl;
    }
    catch (...)
    {
        std::cout << "other exception" << std::endl;
    }

    while (1)
    {
        // printf(".");
    }
}