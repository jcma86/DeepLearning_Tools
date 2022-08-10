#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>

#include <iostream>

typedef websocketpp::client<websocketpp::config::asio_client> client;

using websocketpp::lib::bind;
using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;

typedef websocketpp::config::asio_client::message_type::ptr message_ptr;

void on_message(client *c, websocketpp::connection_hdl hdl, message_ptr msg)
{
	std::cout << "on_message called with hdl: " << hdl.lock().get()
			  << " and message: " << msg->get_payload()
			  << std::endl;

	websocketpp::lib::error_code ec;

	c->send(hdl, msg->get_payload(), msg->get_opcode(), ec);
	if (ec)
	{
		std::cout << "Echo failed because: " << ec.message() << std::endl;
	}
}

bool connected = false;
void on_open(websocketpp::connection_hdl hdl)
{
	printf("Connected Weeeee\n");
	connected = true;
}

void *clientThread(void *params)
{
	client *myClient = (client *)params;

	myClient->run();

	return NULL;
}

int main(int argc, char *argv[])
{
	client myClient;
	client::connection_ptr connection;

	std::string uri = "ws://localhost:9002";

	if (argc == 2)
	{
		uri = argv[1];
	}

	try
	{
		pthread_attr_t attr;
		pthread_t thread;

		pthread_attr_init(&attr);
		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

		myClient.set_access_channels(websocketpp::log::alevel::all);
		myClient.clear_access_channels(websocketpp::log::alevel::frame_payload);
		myClient.init_asio();
		myClient.set_message_handler(bind(&on_message, &myClient, ::_1, ::_2));
		// myClient.set_open_handler(bind(&on_open, &myClient, ::_1, ::_2));
		myClient.set_open_handler(&on_open);

		websocketpp::lib::error_code ec;
		connection = myClient.get_connection(uri, ec);
		if (ec)
		{
			std::cout << "could not create connection because: " << ec.message() << std::endl;
			return 0;
		}

		myClient.connect(connection);

		int rc = pthread_create(&thread, &attr, &clientThread, (void *)&myClient);
		if (rc)
		{
			printf("[ERROR] Couln't initialize WebSocket client.\n");
			exit(-1);
		}
	}
	catch (websocketpp::exception const &e)
	{
		std::cout << e.what() << std::endl;
	}

	bool sent = false;
	while (1)
	{
		if (!sent && connected)
		{
			websocketpp::lib::error_code ec;
			std::string msg = "Hello";
			// message_ptr msg;
			// msg->append_payload("A payload");
			myClient.send(connection->get_handle(), msg, websocketpp::frame::opcode::text, ec);

			sent = true;
		}
	}

	return 0;
}