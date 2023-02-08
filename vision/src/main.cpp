#include <vector>

#include <pigpio.h>

#include "cpp-tools/src/sighandle.h"
#include "cpp-tools/src/timing.h"
//#define OPENCV_TRAITS_ENABLE_DEPRECATED
#include <core/visionserver2.h>
#include <core/visioncamera.h>
#include <core/config.h>
#include <core/neon.h>
#define APRILPOSE_DEBUG
#include <core/aprilpose.h>

#include "rapidreact2.h"
#include "calibrations.h"
#include "field.h"


extern "C" int32_t test_add6_asm(int32_t a, int32_t b, int32_t c, int32_t d, int32_t e, int32_t f);
void perftest();
void featureDemo();
void i2c(bool*);
void i2c_(bool*);
void I2C();

class TestTarget : public vs2::UniqueTarget<TestTarget> {
public:
	inline TestTarget() : UniqueTarget("Test") {}
};

std::thread sl, ma;
bool s = true;
StopWatch runtime("Runtime", &std::cout, 0);
void on_exit() {
	runtime.end();
}

int main(int argc, char* argv[]) {
	runtime.setStart();
	SigHandle::get();
	atexit(on_exit);

	//perftest();

	std::vector<VisionCamera> cameras;

	if(argc > 1 && initNT(argv[1]) && createCameras(cameras, calibrations, argv[1])) {}
	else if(initNT() && createCameras(cameras, calibrations)) {}
	else { return EXIT_FAILURE; }

	vs2::VisionServer::Init();
	vs2::VisionServer::addCameras(std::move(cameras));
	vs2::VisionServer::addStreams(1);
	UHPipeline uh_pipe(vs2::BGR::BLUE);
	CargoPipeline c_pipe;
	vs2::VisionServer::addPipelines({&uh_pipe, &c_pipe});
	AprilPose ap{FIELD_2022};
	vs2::VisionServer::addPipeline(&ap);
	// vs2::VisionServer::compensate();
	// {
	// 	std::vector<TestTarget> targets;
	// 	targets.resize(3);
	// }
	//featureDemo();
	vs2::VisionServer::run(60.f);
	atexit(vs2::VisionServer::stopExit);

	//bool s = true;
	//gpioInitialise();
	// sl = std::thread(i2c, &s);
	// ma = std::thread(i2c_, &s);
	// for(;s;) {
	// 	if(std::cin.get()) {
	// 		s = false;
	// 		sl.join();
	// 		ma.join();
	// 	}
	// 	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	// }
	//I2C();
	//gpioTerminate();

}

// LIST OF THINGS
/*	x = done, x? = kind of done
x Dynamic resizing/scaling
x Position math -> networktables
x Test communication with robot -> target positioning w/ drive program
x multiple cameras -> switching (find out what we want to do)
x compression/stay under bandwidth limit
x Modularize?
x MORE CUSTOM ASSEMBLY!!!
x Target abstraction and generalization (pipeline template param)
x System for telling the robot when targeting info is outdated
x Toggle pipeline processing (processing or just streaming)
x Networktables continuity with multiple class instances
x Multiple VisionServer processing instances, data protection/management -> vector of threads?
- Robot-program mode where all settings are determined by robot program over ntables
- Automatically deduce nt-connection mode
x TensorFlow models
x VS2 Targets/ntables output
x Coral Edge TPU delegate support
- Aruco/AprilTag-specific features?
- >> LOGGING <<
- Improve docs
- Characterize and optimize ftime/threading
- More robust/dynamic config & calibration options
*/



void perftest() {
	constexpr size_t TEST_FRAMES{100};
	const cv::Size fsize{640, 480};

	cv::Mat frame{fsize, CV_8UC3}, binary{fsize, CV_8UC1};
	std::array<cv::Mat, 3> channels{cv::Mat(fsize, CV_8UC1), cv::Mat(fsize, CV_8UC1), cv::Mat(fsize, CV_8UC1)};
	for(size_t i = 0; i < frame.size().area() * frame.channels(); i++) {
		frame.data[i] = rand() % 255;
	}

	std::cout << "FSize: " << fsize << " --> IsContinuous: " << frame.isContinuous() << std::endl;
	cv::imwrite("/mnt/usb0/frame.jpg", frame);

	//cv::split(frame, channels);

	HRC::time_point beg, end;
	beg = HRC::now();
	for(size_t i = 0; i < TEST_FRAMES; i++) {
		cv::split(frame, channels);
		cv::addWeighted(
			channels[2], 0.8,
			channels[0], 0.8,
			50, binary
		);
		cv::subtract(channels[1], binary, binary);
	}
	end = HRC::now();
	std::cout << "CVx" << TEST_FRAMES << " (ms): " << (end - beg).count() / 1e6 << std::endl;
	cv::imwrite("/mnt/usb0/cv.jpg", binary);
	beg = HRC::now();
	for(size_t i = 0; i < TEST_FRAMES; i++) {
		//cv::split(frame, channels);
		// memcpy_wst_asm(
		// 	channels[1].data,
		// 	channels[2].data,
		// 	channels[0].data,
		// 	binary.data,
		// 	fsize.area(),
		// 	(uint8_t)(255 * 0.5),
		// 	(uint8_t)(255 * 0.5),
		// 	50
		// );
		memcpy_split_wst_asm(
			frame.data,
			binary.data,
			fsize.area(),
			1,
			(uint8_t)(255 * 0.8),
			(uint8_t)(255 * 0.8),
			50
		);
	}
	end = HRC::now();
	std::cout << "ASMx" << TEST_FRAMES << " (ms): " << (end - beg).count() / 1e6 << std::endl;
	cv::imwrite("/mnt/usb0/asm.jpg", binary);
}

#include <unistd.h>
#include <termios.h>
char getch(int block = 1) {
	char buf = 0;
	struct termios old = {0};

	if (tcgetattr(0, &old) < 0) {
		perror("tcsetattr()");
	}

	old.c_lflag &= ~ICANON;
	old.c_lflag &= ~ECHO;
	old.c_cc[VMIN] = block;
	old.c_cc[VTIME] = 0;

	if (tcsetattr(0, TCSANOW, &old) < 0) {
		perror("tcsetattr ICANON");
	}
	if (read(0, &buf, 1) < 0) {
		perror ("read()");
	}

	old.c_lflag |= ICANON;
	old.c_lflag |= ECHO;

	if (tcsetattr(0, TCSADRAIN, &old) < 0) {
		perror ("tcsetattr ~ICANON");
	}

	return (buf);
}
void featureDemo() {
	std::cout << "Feature Demo v1.0.0\n[r]: run raw\n[s]: run single\n[m]: run multi\n[q]: stop vs\n[e]: exit\nCameras: "
		<< vs2::VisionServer::numCameras() << "\nPipelines: " << vs2::VisionServer::numPipelines() << "\nStreams: " << vs2::VisionServer::numStreams()
		<< std::endl;
	
	char i[5] = {0}, last[5] = {0};
	int s = 1;
	for(;;) {
		i[0] = getch();
		while(i[s] = getch(0)) {
			s++;
		}
		s = 1;
		switch(i[0]) {
			case 'e': {
				vs2::VisionServer::stop();
				std::cout << "Exiting...\n";
				return;
			}
			case 'q': {
				vs2::VisionServer::stop();
				std::cout << "Stopped VS.\n";
				break;
			}
			case 'r': {
				std::cout << (vs2::VisionServer::runRawThread() ? "Running raw.\n" : "Unable to run raw; stop previously running instances.\n");
				break;
			}
			case 's': {
				std::cout << (vs2::VisionServer::runSingleThread() ? "Running single.\n" : "Unable to run single; stop previously running instances.\n");
				break;
			}
			case 'm': {
				std::cout << (vs2::VisionServer::runThread() ? "Running multi.\n" : "Unable to run multi; stop previously running instances.\n");
				break;
			}
			case 0: break;
			default: {
				std::cout << "Invalid Input.\n";
			}
		}
		std::cout.flush();

	}

}



#include <pigpio.h>

void i2c(bool* s) {
#define RE	(1 << 9)	// recieve enable
#define TE	(1 << 8)	// transmit enable
#define BK	(1 << 7)	// break and clear buffers
#define I2E	(1 << 2)	// enable i2c mode
#define EN	(1 << 0)	// enable bsc

	bsc_xfer_t transfer;
	//gpioInitialise();
	transfer.control = ((0x08 << 16) | RE | TE | I2E | EN);

	if(bscXfer(&transfer) >= 0) {
		transfer.rxCnt = 0;
		std::cout << "Beginning Slave Transfer Queue" << std::endl;
		int r_t = 0;
		for(;*s;) {
			bscXfer(&transfer);
			if(transfer.rxCnt > 0) {
				while(transfer.rxCnt > 0) {
					r_t += transfer.rxCnt;
					std::cout << "\nSlave recieved " << std::dec << transfer.rxCnt << " bytes.(Total:" << r_t << ")\n\t" << std::hex;
					for(int i = 0; i < transfer.rxCnt; i++) {
						std::cout << "0x" << (int)transfer.rxBuf[i] << "  ";
					}
					transfer.rxCnt = 0;
					bscXfer(&transfer);
				}
				(std::cout << "\n\n").flush();
			}
			// if(transfer.rxCnt > 0) {
			// 	std::cout << "Recieved " << transfer.rxCnt << " bytes:\n\t"
			// 		<< *((float*)transfer.rxBuf) << std::endl;
				
			// 	// const char* mss = "Return Message from 0x08!";
			// 	// strcpy(transfer.txBuf, mss);
			// 	// transfer.txCnt = strlen(mss);
			// }
			//if(std::cin.get()) { break; }
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
	}
	transfer.control = ((0x08 << 16) | BK | !I2E | !EN);
	bscXfer(&transfer);
	//gpioTerminate();

}
void i2c_(bool* s) {
	//gpioInitialise();
	int h = i2cOpen(1, 0x08, 0);
	if(h >= 0) {
		char buff[32];
		char buff2[32];
		//strcpy(buff, "Random Number: ");
		//buff[18] = '\0';
		int32_t itr = 1;
		const int sz = 16;
		for(;*s;) {
			std::cout << "Sending Bytes: " << std::dec << sz << " (Total: " << ((itr++) * sz) << ")\n\t" << std::hex;
			for(int i = 0; i < sz; i++) {
				buff[i] = rand() & 0xFF;
				std::cout << (int)buff[i] << " ";
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
			if(i2cWriteDevice(h, buff, sz)) {
				std::cout << " --> Write Error";
			}
			std::cout << std::endl;
			//int r = rand() & 0xFF;
			//buff[15] = (r / 100) + '0';
			//buff[16] = ((r % 100) / 10) + '0';
			//buff[17] = (r % 10) + '0';
			//i2cWriteDevice(h, buff, 19);
			// int i = 0, t;
			// while(((t = i2cReadDevice(h, buff2, 32)) > 0) && i < 10) {
			// 	for(int k = 0; k < t; k++) {
			// 		std::cout << buff2[k];
			// 	}
			// 	i += (t + 1);
			// 	std::this_thread::sleep_for(std::chrono::milliseconds(10));
			// }
			// memset(buff2, 0, 32);
			// std::cout << std::endl;
			//if(std::cin.get()) { break; }
			std::this_thread::sleep_for(std::chrono::milliseconds(40));
		}
	}
	i2cClose(h);
	//gpioTerminate();
}

inline int xfer_transmitted(int s) { return (s >> 16) & 0b11111; }
inline int xfer_rfifo(int s) { return (s >> 11) & 0b11111; }
inline int xfer_tfifo(int s) { return (s >> 6) & 0b11111; }
void I2C() {
	const int id = 0x08;
	bsc_xfer_t xfer;
	xfer.control = ((id << 16) | RE | TE | I2E | EN);
	int h = i2cOpen(1, id, 0);
	if(h >= 0 && bscXfer(&xfer) >= 0) {
		//uint8_t buff[32];
		memset(xfer.txBuf, 0, 512);
		bscXfer(&xfer);
		bscXfer(&xfer);
		std::cout << "Randomizing TxBuffer:\n" << std::hex;
		for(int i = 0; i < sizeof(xfer.txBuf) / sizeof(int); i++) {
			((int*)xfer.txBuf)[i] = rand();
			std::cout << "0x" << ((int*)xfer.txBuf)[i] << ' ';
		}
		std::cout << "\n\n" << std::dec;
		xfer.txCnt = sizeof(xfer.txBuf);
		int k = bscXfer(&xfer);
		std::cout << "Copied " << xfer_transmitted(k) << " bytes, "
			<< xfer_rfifo(k) << " bytes in recieve FIFO, " << xfer_tfifo(k) << " bytes in transmit fifo\n"
			<< "Is transmit full?: " << ((k >> 2) & 1) << "\nFull status: " << k << "\n\n";
		// while(k - xfer_transmitted(bscXfer(&xfer))) {
		// }
		uint8_t buff[32];
		std::cout << "Buffer before read:\n" << std::hex;
		for(int i = 0; i < sizeof(buff) / sizeof(int); i++) {
			std::cout << "0x" << ((int*)buff)[i] << ' ';
		}
		k = i2cReadDevice(h, (char*)buff, sizeof(buff));
		std::cout << std::dec << "\n\nRead " << k << " bytes\n" << std::hex;
		for(int i = 0; i < sizeof(buff) / sizeof(int); i++) {
			std::cout << "0x" << ((int*)buff)[i] << ' ';
		}
		std::cout << std::endl;


		//memset(buff, 0, sizeof(buff));
		memcpy(buff, "fjslfdkskjflasdjfkdsjlkdjfdsewe", 32);
		std::cout << "\n\nWrite buffer:\n";
		for(int i = 0; i < sizeof(buff); i++) {
			std::cout << "0x" << (int)buff[i] << ' ';
		}
		std::cout << std::dec << "\nWrite status: " << i2cWriteDevice(h, (char*)buff, 32) << "\n\n";

		k = bscXfer(&xfer);
		std::cout << "Recieved " << xfer.rxCnt << " bytes, "
			<< xfer_rfifo(k) << " bytes in recieve FIFO, " << xfer_tfifo(k) << " bytes in transmit fifo\n"
			<< "Is transmit full?: " << ((k >> 2) & 1) << "\nFull status: " << k << "\n\n" << std::hex;
		for(int i = 0; i < xfer.rxCnt; i++) {
			std::cout << "0x" << (int)xfer.rxBuf[i] << ' ';
		}
		std::cout << std::endl;


		std::cout << std::dec << "\n\nRead at 0x00: " << i2cReadI2CBlockData(h, 0x00, (char*)buff, 32);
		k = bscXfer(&xfer);
		std::cout << "\n\nRecieved " << xfer.rxCnt << " bytes, "
			<< xfer_rfifo(k) << " bytes in recieve FIFO, " << xfer_tfifo(k) << " bytes in transmit fifo\n"
			<< "Is transmit full?: " << ((k >> 2) & 1) << "\nFull status: " << k << "\n\n" << std::hex;
		for(int i = 0; i < xfer.rxCnt; i++) {
			std::cout << "0x" << (int)xfer.rxBuf[i] << ' ';
		}
		std::cout << std::dec << "\n\nWrite at 0x00: " << i2cWriteBlockData(h, 0x00, (char*)buff, 1) << ", First byte: 0x" << std::hex << (int)buff[0] << std::dec;
		k = bscXfer(&xfer);
		std::cout << "\n\nRecieved " << xfer.rxCnt << " bytes, "
			<< xfer_rfifo(k) << " bytes in recieve FIFO, " << xfer_tfifo(k) << " bytes in transmit fifo\n"
			<< "Is transmit full?: " << ((k >> 2) & 1) << "\nFull status: " << k << "\n\n" << std::hex;
		for(int i = 0; i < xfer.rxCnt; i++) {
			std::cout << "0x" << (int)xfer.rxBuf[i] << ' ';
		}
		//std::cout << std::dec << "\n\nRead at 0x00: " << i2cReadBlockData(h, 0x00, (char*)buff);

		// for(;;) {
		// 	int k = i2cReadDevice(h, xfer.rxBuf, sizeof(xfer.txBuf));
		// }
	}
	xfer.control = ((id << 16) | BK | !I2E | !EN);
	bscXfer(&xfer);
	i2cClose(h);
}