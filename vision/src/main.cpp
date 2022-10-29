#include <vector>

#include "cpp-tools/src/sighandle.h"
#include "cpp-tools/src/timing.h"
//#define OPENCV_TRAITS_ENABLE_DEPRECATED
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

class ArucoTest;

StopWatch runtime("Runtime", &std::cout, 0);
void on_exit() { runtime.end(); }

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
	//UHPipeline uh_pipe(vs2::BGR::BLUE);
	//CargoPipeline c_pipe;
	//vs2::VisionServer::addPipelines({&uh_pipe, &c_pipe});
	vs2::VisionServer::addPipeline<ArucoTest>();
	vs2::VisionServer::compensate();
	vs2::VisionServer::run(60);
	atexit(vs2::VisionServer::stopExit);

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



// All of this will eventually be moved to a separate file, and possibly part of it into the main library
#include <opencv2/aruco.hpp>
#include <core/vision.h>

// https://docs.opencv.org/4.5.2/db/da9/tutorial_aruco_board_detection.html
// https://docs.google.com/document/d/e/2PACX-1vQxVFxsY30_6sy50N8wWUpUhQ0qbUKnw7SjW6agbKQZ2X0SN_uXtNZhLB7AkRcJjLnlcmmjcyCNhn0I/pub
// https://docs.google.com/document/d/e/2PACX-1vQxVFxsY30_6sy50N8wWUpUhQ0qbUKnw7SjW6agbKQZ2X0SN_uXtNZhLB7AkRcJjLnlcmmjcyCNhn0I/pub
// https://docs.google.com/document/d/e/2PACX-1vSizkGFRocq8-QLCj38O68MO4wYCThk_z60g7KhBLf497UqnLHcLW9r1HcTKzwI_SoYLHZp7wPnU6H4/pub
/*
	Apparently the cv::aruco::Board class can be used to represent any 3D map of markers, so this
	should act as a very nice way to implement a field-position detection pipeline. We just
	need to get all of the locations of the markers as well as their ID's, then create
	a board object.
*/
class ArucoTest : public vs2::VPipeline<ArucoTest> {
public:
	inline ArucoTest() : VPipeline("Aruco Test Pipeline") {}
	void process(cv::Mat& io_frame) override {
		this->corners.clear();
		this->ids.clear();
		cv::Size2i fsz = io_frame.size() / (int)SCALE;
		if(this->buffer.size() != fsz) {
			this->buffer = cv::Mat(fsz, CV_8UC3);
		}
		cv::resize(io_frame, this->buffer, fsz);
		cv::aruco::detectMarkers(this->buffer, this->markers, this->corners, this->ids);
		if(this->corners.size() > 0) {
			rescale(this->corners, SCALE);
			if(cv::aruco::estimatePoseBoard(
				this->corners, this->ids, FIELD_2022,
				this->getSrcMatrix(), this->getSrcDistort(),
				this->rvec, this->tvec
			)) {
				cv::aruco::drawAxis(io_frame, this->getSrcMatrix(), this->getSrcDistort(), this->rvec, this->tvec, 100.f);
			}
			cv::aruco::drawDetectedMarkers(io_frame, this->corners, this->ids);
		}
	}


	constexpr static inline cv::aruco::PREDEFINED_DICTIONARY_NAME
		FRC_DICT = cv::aruco::DICT_APRILTAG_36h11,
		SIMPLE_DICT = cv::aruco::DICT_4X4_50
	;
	constexpr static inline size_t
		SCALE{2};
	// constexpr static inline bool
	// 	REFINE_DETECTIONS = true;

	static inline const cv::Ptr<cv::aruco::Board>	// the 2022 (nonofficial) field has 24 markers - link to source above
		FIELD_2022{ cv::aruco::Board::create(
			// calculations and formatting: https://docs.google.com/spreadsheets/d/1zpj37KxVP_r6md_0VjLPmQ9h4aVExXe4bx78hdGrro8/edit?usp=sharing
			std::vector<std::vector<cv::Point3f>>{ {
				{
					cv::Point3f(-0.139,295.133,38.126),
					cv::Point3f(-0.139,301.633,38.126),
					cv::Point3f(-0.139,301.633,31.626),
					cv::Point3f(-0.139,295.133,31.626)
				}, {
					cv::Point3f(127.272,212.76,71.182),
					cv::Point3f(127.272,219.26,71.182),
					cv::Point3f(127.272,219.26,64.682),
					cv::Point3f(127.272,212.76,64.682)
				}, {
					cv::Point3f(117.53,209.863,57.432),
					cv::Point3f(124.03,209.863,57.432),
					cv::Point3f(124.03,209.863,50.932),
					cv::Point3f(117.53,209.863,50.932)
				}, {
					cv::Point3f(0.157,195.905,35),
					cv::Point3f(0.157,202.405,35),
					cv::Point3f(0.157,202.405,28.5),
					cv::Point3f(0.157,195.905,28.5)
				}, {
					cv::Point3f(0.157,135.037,35),
					cv::Point3f(0.157,141.537,35),
					cv::Point3f(0.157,141.537,28.5),
					cv::Point3f(0.157,135.037,28.5)
				}, {
					cv::Point3f(7.11568287669421,65.3835825687076,38.313),
					cv::Point3f(2.42031712330579,69.8784174312924,38.313),
					cv::Point3f(2.42031712330579,69.8784174312924,31.813),
					cv::Point3f(7.11568287669421,65.3835825687076,31.813)
				}, {
					cv::Point3f(36.7296828766942,34.8115825687076,38.313),
					cv::Point3f(32.0343171233058,39.3064174312924,38.313),
					cv::Point3f(32.0343171233058,39.3064174312924,31.813),
					cv::Point3f(36.7296828766942,34.8115825687076,31.813)
				}, {
					cv::Point3f(65.9336828766942,3.94358256870762,38.313),
					cv::Point3f(61.2383171233058,8.43841743129238,38.313),
					cv::Point3f(61.2383171233058,8.43841743129238,31.813),
					cv::Point3f(65.9336828766942,3.94358256870762,31.813)
				}, {
					cv::Point3f(648.139,28.867,38.126),
					cv::Point3f(648.139,22.367,38.126),
					cv::Point3f(648.139,22.367,31.626),
					cv::Point3f(648.139,28.867,31.626)
				}, {
					cv::Point3f(521.063,111.26,71.182),
					cv::Point3f(521.063,104.76,71.182),
					cv::Point3f(521.063,104.76,64.682),
					cv::Point3f(521.063,111.26,64.682)
				}, {
					cv::Point3f(530.47,114.167,57.432),
					cv::Point3f(523.97,114.167,57.432),
					cv::Point3f(523.97,114.167,50.932),
					cv::Point3f(530.47,114.167,50.932)
				}, {
					cv::Point3f(647.843,128.27,35),
					cv::Point3f(647.843,121.77,35),
					cv::Point3f(647.843,121.77,28.5),
					cv::Point3f(647.843,128.27,28.5)
				}, {
					cv::Point3f(647.843,188.964,35),
					cv::Point3f(647.843,182.464,35),
					cv::Point3f(647.843,182.464,28.5),
					cv::Point3f(647.843,188.964,28.5)
				}, {
					cv::Point3f(640.861534684921,258.84072074132,38.438),
					cv::Point3f(645.360465315079,254.14927925868,38.438),
					cv::Point3f(645.360465315079,254.14927925868,31.938),
					cv::Point3f(640.861534684921,258.84072074132,31.938)
				}, {
					cv::Point3f(611.549534684921,289.45972074132,38.313),
					cv::Point3f(616.048465315079,284.76827925868,38.313),
					cv::Point3f(616.048465315079,284.76827925868,31.813),
					cv::Point3f(611.549534684921,289.45972074132,31.813)
				}, {
					cv::Point3f(582.285534684921,320.02772074132,38.313),
					cv::Point3f(586.784465315079,315.33627925868,38.313),
					cv::Point3f(586.784465315079,315.33627925868,31.813),
					cv::Point3f(582.285534684921,320.02772074132,31.813)
				}, {
					cv::Point3f(312.974022737338,194.753894089996,30.938),
					cv::Point3f(307.035977262662,192.110105910004,30.938),
					cv::Point3f(307.035977262662,192.110105910004,24.438),
					cv::Point3f(312.974022737338,194.753894089996,24.438)
				}, {
					cv::Point3f(291.246105910004,150.974022737338,30.938),
					cv::Point3f(293.889894089996,145.035977262662,30.938),
					cv::Point3f(293.889894089996,145.035977262662,24.438),
					cv::Point3f(291.246105910004,150.974022737338,24.438)
				}, {
					cv::Point3f(335.025977262662,129.246105910004,30.938),
					cv::Point3f(340.964022737338,131.889894089996,30.938),
					cv::Point3f(340.964022737338,131.889894089996,24.438),
					cv::Point3f(335.025977262662,129.246105910004,24.438)
				}, {
					cv::Point3f(356.753894089996,173.025977262662,30.938),
					cv::Point3f(354.110105910004,178.964022737338,30.938),
					cv::Point3f(354.110105910004,178.964022737338,24.438),
					cv::Point3f(356.753894089996,173.025977262662,24.438)
				}, {
					cv::Point3f(302.123035778737,173.879364166192,98.0881815660862),
					cv::Point3f(299.793644106692,167.81109139396,98.0881815660862),
					cv::Point3f(302.524964221264,166.762635833808,92.2838184339138),
					cv::Point3f(304.854355893308,172.83090860604,92.2838184339138)
				}, {
					cv::Point3f(312.120635833808,140.123035778737,98.0881815660862),
					cv::Point3f(318.18890860604,137.793644106692,98.0881815660862),
					cv::Point3f(319.237364166192,140.524964221264,92.2838184339138),
					cv::Point3f(313.16909139396,142.854355893308,92.2838184339138)
				}, {
					cv::Point3f(345.876964221264,150.120635833808,98.0881815660862),
					cv::Point3f(348.206355893308,156.18890860604,98.0881815660862),
					cv::Point3f(345.475035778737,157.237364166192,92.2838184339138),
					cv::Point3f(343.145644106692,151.169091393961,92.2838184339138)
				}, {
					cv::Point3f(335.879364166192,183.876964221263,98.0881815660862),
					cv::Point3f(329.811091393961,186.206355893308,98.0881815660862),
					cv::Point3f(328.762635833808,183.475035778737,92.2838184339138),
					cv::Point3f(334.83090860604,181.145644106692,92.2838184339138)
				}
			} },
			cv::aruco::getPredefinedDictionary(FRC_DICT),
			std::array<int32_t, 24>{
				0, 1, 2, 3, 4, 5, 6, 7,
				10, 11, 12, 13, 14, 15, 16, 17,
				40, 41, 42, 43,
				50, 51, 52, 53
			}
		)};

protected:
	cv::Ptr<cv::aruco::Dictionary> markers{
		cv::aruco::getPredefinedDictionary(FRC_DICT)
	};

	std::vector<std::vector<cv::Point2f> > corners;
	std::vector<int32_t> ids;
	std::array<float, 3> tvec, rvec;
	cv::Mat buffer;


};