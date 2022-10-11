#pragma once

#include <vector>
#include <array>
#include <thread>
#include <string>
#include <type_traits>

#include <opencv4/opencv2/opencv.hpp>

#include "core/visionserver2.h"
#include "core/extensions.h"
#include "core/target.h"
#include "core/mem.h"


// UPPER HUB DETECTION
// TODO:
/*
 - update rotational angle derivation (primarily U/D)
*/

class UHPipeline : public vs2::VPipeline<UHPipeline> {
public:
	UHPipeline(vs2::BGR c = vs2::BGR::GREEN);
	UHPipeline(const UHPipeline&) = delete;

	virtual void process(cv::Mat& io_frame) override;

private:	// currently the starting size is 848 bytes
// thresholding
	std::array<cv::Mat, 3> channels;
	cv::Mat binary, buffer;
	double alpha{0.5}, beta{0.5}, gamma{0.0};	// add ntable options
	uint8_t thresh{50};
	size_t scale{1};
	vs2::BGR color;
// contours
	std::vector<std::vector<cv::Point2i> > contours;
	double area_largest{0.f}, area_buff{0.f};
	int16_t target_idx{-1};
// algorithm
	std::vector<cv::Rect> in_range;
	cv::Rect rect_buff;
	cv::Size range_buff;
	struct UpperHub : public vs2::Target {
	friend class UHPipeline;
		inline static constexpr size_t
			MIN_DETECT = 4,
			MAX_DETECT = 6;
		inline static const std::array<cv::Point3f, 6> world_coords{	// left to right
			cv::Point3f(0.f, 103.f, 26.73803044f),				// @c=0
			cv::Point3f(10.23220126f, 103.f, 24.70271906f),		// @c=10.5
			cv::Point3f(18.90664264f, 103.f, 18.90664264f),		// @c=21
			cv::Point3f(24.70271906f, 103.f, 10.23220126f),		// @c=31.5
			cv::Point3f(26.73803044f, 103.f, 0.f),				// @c=42
			cv::Point3f(24.70271906f, 103.f, -10.23220126f),	// @c=52.5
		};
		
		inline UpperHub() : Target("Upper Hub") {}
		UpperHub(const UpperHub&) = delete;
	} target;
// pose solving
	std::vector<cv::Point2f> points_buff;
	std::vector<cv::Point3f> world_buff;
	std::array<cv::Mat, 2> tvecs, rvecs;
	cv::Mat_<float>
		rvec = cv::Mat_<float>(1, 3),
		tvec = rvec
	;


};





enum class CargoColor {
	NONE = 0b00,
	RED = 0b01,
	BLUE = 0b10,
	BOTH = 0b11
};
template<typename t = uint8_t>
inline t operator~(CargoColor c) { return static_cast<t>(c); }


template<typename t = float>
struct CvCargo_ {
	inline CvCargo_() {
		static_assert(std::is_arithmetic<t>::value, "Template paramter t must be arithmetic type.");
	}
	inline CvCargo_(cv::Point_<t> c, t r) : center(c), radius(r) {}
	inline CvCargo_(cv::Point_<t> c, t r, CargoColor color) : center(c), radius(r), color(color) {}

	cv::Point_<t> center;
	t radius{(t)0.0};
	CargoColor color{CargoColor::NONE};

	inline const CvCargo_<t>& rescale(double s) {
		this->center *= s;
		this->radius *= s;
		return *this;
	}
	inline const CvCargo_<t>& operator*=(double s) { return this->rescale(s); }

	inline bool operator<(const CvCargo_<t>& c) { return this->radius < c.radius; }		// compares relative size
	inline bool operator>(const CvCargo_<t>& c) { return this->radius > c.radius; }		// '''

};
typedef CvCargo_<>	CvCargo;


template<CargoColor color = CargoColor::NONE>
class Cargo : public vs2::UniqueTarget<Cargo<color>> {
	friend class CargoPipeline;
	typedef struct Cargo<color>		This_t;
public:
	inline static const std::array<cv::Point3f, 4> world_coords{
		cv::Point3f(-4.75f, 0.f, 0.f),
		cv::Point3f(0.f, 4.75f, 0.f),
		cv::Point3f(4.75f, 0.f, 0.f),
		cv::Point3f(0.f, -4.75f, 0.f)
	};
	inline static const std::array<const char*, 4> name_map{
		"Unkn Cargo #",
		"Red Cargo #",
		"Blue Cargo #",
		"Rndm Cargo #"
	};

	Cargo() : vs2::UniqueTarget<This_t>(name_map[~color]) {}
	Cargo(const Cargo&) = delete;
	Cargo(Cargo&&) = default;

	template<typename t = float>
	void update(const CvCargo_<t>& v, cv::InputArray matx, cv::InputArray dist);


};
typedef Cargo<CargoColor::RED>	RedCargo;
typedef Cargo<CargoColor::BLUE>	BlueCargo;



class CargoPipeline : public vs2::VPipeline<CargoPipeline> {
public:
	CargoPipeline();
	CargoPipeline(const CargoPipeline&) = delete;

	void process(cv::Mat& io_frame) override;

protected:
	template<vs2::BGR base, int a = 50, int b = 50, int g = 0, int tr = 30>
	class CargoFilter {
		friend class CargoPipeline;
	public:
		void threshold(const std::array<cv::Mat, 3>& channels);

	protected:
		constexpr inline static float
			CONTOUR_AREA_THRESH = 500.f,
			MIN_CIRCULARITY = 0.8f;

		cv::Mat binary;
		std::vector<std::vector<cv::Point2i> > contours;
		std::vector<cv::Point2i> point_buff;
		double
			alpha{a / 100.0},
			beta{b / 100.0},
			gamma{g / 100.0},
			thresh{tr / 100.0};

		std::vector<CvCargo> objects;

	};

	std::array<cv::Mat, 3> channels;
	cv::Mat buffer;
	size_t scale{1};

	template<
		CargoColor key,
		vs2::BGR base,
		int a, int b, int g, int tr
	> struct CargoPair {
		constexpr static inline vs2::BGR B_CLR{ base };

		std::vector<Cargo<key> > targets;
		CargoFilter<base, a, b, g, tr> proc;
	};

	CargoPair<CargoColor::RED, vs2::BGR::RED, 100, 100, 0, 30> red;
	CargoPair<CargoColor::BLUE, vs2::BGR::BLUE, 20, 100, 0, 15> blue;


};











template<CargoColor color>
template<typename t>
void Cargo<color>::update(const CvCargo_<t>& v, cv::InputArray matx, cv::InputArray dist) {
	cv::Mat1f tvec, rvec;
	std::array<cv::Point2f, 4> outline{
		cv::Point2f(v.center.x - v.radius, v.center.y),
		cv::Point2f(v.center.x, v.center.y - v.radius),
		cv::Point2f(v.center.x + v.radius, v.center.y),
		cv::Point2f(v.center.x, v.center.y + v.radius)
	};
	cv::solvePnP(
		this->world_coords, outline,
		matx, dist, rvec, tvec
	);
	// something happening here >> (crashing w/ 100% thread utilization)
	this->setPos(tvec[0][0], tvec[1][0], tvec[2][0]);
	this->setAngle(
		atan2(tvec[1][0], tvec[2][0]) * -180/CV_PI,
		atan2(tvec[0][0], tvec[2][0]) * -180/CV_PI
	);
	this->setDist(
		sqrt(pow(tvec[0][0], 2) + pow(tvec[1][0], 2) + pow(tvec[2][0], 2))
	);
	//this->setValid();
}

template<vs2::BGR base, int a, int b, int g, int tr>
void CargoPipeline::CargoFilter<base, a, b, g, tr>::threshold(const std::array<cv::Mat, 3>& channels) {

	cv::addWeighted(
		channels[vs2::weights_map[~base][0]], this->alpha,
		channels[vs2::weights_map[~base][1]], this->beta,
		this->gamma, this->binary
	);
	cv::subtract(channels[~base], this->binary, this->binary);
	double maxv;
	cv::minMaxIdx(this->binary, nullptr, &maxv);
	if((int)maxv && (int)this->thresh) {
		memcpy_threshold_asm(
			this->binary.data, this->binary.data,
			this->binary.size().area(), maxv * this->thresh
		);
	}

	this->contours.clear();
	cv::findContours(this->binary, this->contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	this->objects.clear();

	cv::Point2f _center;
	float _radius;
	for(size_t i = 0; i < this->contours.size(); i++) {
		if(cv::contourArea(this->contours[i]) > this->binary.size().area() / CONTOUR_AREA_THRESH) {
			cv::minEnclosingCircle(this->contours[i], _center, _radius);
			cv::convexHull(this->contours[i], this->point_buff);
			if(cv::contourArea(this->point_buff) / (CV_PI * pow(_radius, 2)) > MIN_CIRCULARITY) {
				this->objects.emplace_back(_center, _radius);
			}
			// draw contours
		}
	}

}