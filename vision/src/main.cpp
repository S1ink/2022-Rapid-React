#include <vector>

#include "cpp-tools/src/sighandle.h"
#include "cpp-tools/src/timing.h"

#include <core/visionserver2.h>
#include <core/visioncamera.h>
#include <core/config.h>
#include <core/mem.h>

#include "rapidreact2.h"
#include "calibrations.h"


extern "C" int32_t test_add6_asm(
	int32_t a,
	int32_t b,
	int32_t c,
	int32_t d,
	int32_t e,
	int32_t f
);

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

StopWatch runtime("Runtime", &std::cout, 0);
void on_exit() { runtime.end(); }

int main(int argc, char* argv[]) {
	runtime.setStart();
	SigHandle::get();
	atexit(on_exit);

	perftest();

	// std::vector<VisionCamera> cameras;

	// if(argc > 1 && initNT(argv[1]) && createCameras(cameras, calibrations, argv[1])) {}
	// else if(initNT() && createCameras(cameras, calibrations)) {}
	// else { return EXIT_FAILURE; }

	// vs2::VisionServer::Init();
	// vs2::VisionServer::addCameras(std::move(cameras));
	// vs2::VisionServer::addStreams(1);
	// UHPipeline uh_pipe(vs2::BGR::BLUE);
	// CargoPipeline c_pipe;
	// vs2::VisionServer::addPipelines({&uh_pipe, &c_pipe});
	// vs2::VisionServer::compensate();
	// vs2::VisionServer::run(60);
	// atexit(vs2::VisionServer::stopExit);

}

// LIST OF THINGS
/*	x = done, x? = kind of done
x Dynamic resizing/scaling
x Position math -> networktables
x Test communication with robot -> target positioning w/ drive program
x multiple cameras -> switching (find out what we want to do)
x compression/stay under bandwidth limit
x Modularize?
? MORE CUSTOM ASSEMBLY!!! :)		<--
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