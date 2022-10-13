#include "rapidreact2.h"

#include <algorithm>

#include "core/vision.h"
#include "core/mem.h"


UHPipeline::UHPipeline(vs2::BGR c) :
	vs2::VPipeline<UHPipeline>("Upper Hub Pipeline "), color(c)
{
	this->getTable()->PutBoolean("Output Threshold", false);
}
void UHPipeline::process(cv::Mat& io_frame) {

	if(io_frame.size() != this->buffer.size()*(int)this->scale) {	// resize buffers if change in dimensions
		const cv::Size2i nsz = io_frame.size()/this->scale;
		this->buffer = cv::Mat(nsz, CV_8UC3);
		this->binary = cv::Mat(nsz, CV_8UC1);
		for(size_t i = 0; i < this->channels.size(); i++) {
			this->channels.at(i) = cv::Mat(nsz, CV_8UC1);
		}
	}

	cv::resize(io_frame, this->buffer, {}, 1.0/this->scale, 1.0/this->scale);	// downscale image
	cv::split(this->buffer, this->channels);									// split into channels
	cv::addWeighted(															// add weights to out-colors
		this->channels[vs2::weights_map[~this->color][0]], this->alpha,
		this->channels[vs2::weights_map[~this->color][1]], this->beta,
		this->gamma, this->binary
	);
	cv::subtract(this->channels[~this->color], this->binary, this->binary);
	memcpy_threshold_binary_asm(this->binary.data, this->binary.data, this->binary.size().area(), this->thresh);

	this->contours.clear();
	cv::findContours(this->binary, this->contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	if(this->getTable()->GetBoolean("Output Threshold", false)) {
		cv::cvtColor(this->binary, this->buffer, cv::COLOR_GRAY2BGR, 3);
		cv::resize(this->buffer, io_frame, {}, this->scale, this->scale, cv::INTER_NEAREST);
	}

	size_t reinserted = 0;
	this->in_range.clear();
	for(size_t i = 0; i < this->contours.size(); i++) {
		::rescale(this->contours.at(i), this->scale);
	}

	for(size_t i = 0; i < this->contours.size(); i++) {

		this->rect_buff = cv::boundingRect(this->contours.at(i));
		if(::inRange(((float)this->rect_buff.width / this->rect_buff.height), 1.75f, 2.75f)) {
			
			this->in_range.push_back(this->rect_buff);
			this->range_buff = cv::Size(this->rect_buff.width * 7, this->rect_buff.height * 4);
			
			std::iter_swap(this->contours.begin() + reinserted, this->contours.begin() + i);
			reinserted++;

				cv::rectangle(io_frame, this->rect_buff, {0, 255, 255});	// debug

			for(size_t n = reinserted; n < this->contours.size(); n++) {

				this->rect_buff = cv::boundingRect(this->contours.at(n));
				if(::distance(this->rect_buff.tl(), this->in_range[0].tl()) <= this->range_buff) {

					float ratio = (float)this->rect_buff.width / this->rect_buff.height;
					if(::inRange(ratio, 1.75f, 2.75f) && this->rect_buff.area() > this->in_range[0].area()) {

						this->in_range.insert(this->in_range.begin(), this->rect_buff);
						this->range_buff = cv::Size(this->rect_buff.width * 6, this->rect_buff.height * 3);

						std::iter_swap(this->contours.begin() + reinserted, this->contours.begin() + n);
						reinserted++;

							cv::rectangle(io_frame, this->rect_buff, {0, 255, 0});	// debug

					} else if(
					::inRange(ratio, 0.75f, 3.f) &&
					::inRange((float)this->rect_buff.area(), 0.5f * this->in_range[0].area(), 2.f * this->in_range[0].area())
					) {
						this->in_range.push_back(this->rect_buff);

							cv::rectangle(io_frame, this->rect_buff, {255, 255, 0});	// debug

					}
				}
			}
			if(::inRange(this->in_range.size(), UpperHub::MIN_DETECT, UpperHub::MAX_DETECT)) {

				this->points_buff.clear();
				for(size_t m = 0; m < this->in_range.size(); m++) {
					this->points_buff.emplace_back(::findCenter<float, int>(this->in_range[m]));
				}
				std::sort(
					this->points_buff.begin(),
					this->points_buff.end(),
					[](const cv::Point2f& a, const cv::Point2f& b){ return a.x < b.x; }
				);
				this->world_buff.clear();
				this->world_buff.insert(
					this->world_buff.begin(),
					UpperHub::world_coords.begin(),
					UpperHub::world_coords.begin() + this->points_buff.size()
				);
				cv::solvePnPGeneric(
					this->world_buff, this->points_buff,
					this->getSrcMatrix(), this->getSrcDistort(),
					this->rvecs, this->tvecs,
					false, cv::SOLVEPNP_IPPE
				);
				int best =
					(this->tvecs[0].at<double>({1, 0}) + rvecs[0].at<double>({0, 0}))
					> (this->tvecs[1].at<double>({1, 0}) + rvecs[1].at<double>({0, 0}))
					? 0 : 1;
				this->tvec = this->tvecs[best];
				this->rvec = this->rvecs[best];
				cv::solvePnPRefineLM(
					this->world_buff, this->points_buff,
					this->getSrcMatrix(), this->getSrcDistort(),
					this->rvec, this->tvec
				);
				this->target.setPos(
					this->tvec[0][0],
					this->tvec[1][0],
					this->tvec[2][0]
				);
				this->target.setDist(
					sqrt(pow(this->tvec[0][0], 2) + pow(this->tvec[1][0], 2) + pow(this->tvec[2][0], 2))
				);
				this->target.setAngle(
					0.0,
					acos(this->tvec[2][0] / sqrt(pow(this->tvec[0][0], 2) + pow(this->tvec[2][0], 2)))
						* 180 / CV_PI * sgn(this->tvec[0][0])
				);

				// draw 3d debug

				break;

			} else {
				this->in_range.clear();
				if(i + 1 < reinserted) {
					i = reinserted - 1;
				}
			}
		} else {

				cv::rectangle(io_frame, this->rect_buff, {0, 0, 255});	// debug

		}
	}

}





CargoPipeline::CargoPipeline() :
	VPipeline("Cargo Pipeline ")
{
	this->getTable()->PutBoolean("Process Red", true);
	this->getTable()->GetSubTable("Red Filtering")->PutNumber("Alpha", this->red.proc.alpha);
	this->getTable()->GetSubTable("Red Filtering")->PutNumber("Beta", this->red.proc.beta);
	this->getTable()->GetSubTable("Red Filtering")->PutNumber("Gamma", this->red.proc.gamma);
	this->getTable()->GetSubTable("Red Filtering")->PutNumber("Thresh", this->red.proc.thresh);
	this->getTable()->PutBoolean("Process Blue", true);
	this->getTable()->GetSubTable("Blue Filtering")->PutNumber("Alpha", this->blue.proc.alpha);
	this->getTable()->GetSubTable("Blue Filtering")->PutNumber("Beta", this->blue.proc.beta);
	this->getTable()->GetSubTable("Blue Filtering")->PutNumber("Gamma", this->blue.proc.gamma);
	this->getTable()->GetSubTable("Blue Filtering")->PutNumber("Thresh", this->blue.proc.thresh);
	this->getTable()->PutBoolean("Show Threshold", false);
	this->getTable()->PutBoolean("Show Contours", false);
	this->getTable()->PutNumber("Scaling", 1.0);
}
void CargoPipeline::process(cv::Mat& io_frame) {
	bool
		_red = this->getTable()->GetBoolean("Process Red", false),
		_blue = this->getTable()->GetBoolean("Process Blue", false),
		_contours = this->getTable()->GetBoolean("Show Contours", false),
		_threshold = this->getTable()->GetBoolean("Show Threshold", false);

	this->scale = this->getTable()->GetNumber("Scaling", 1.0);
	if(this->scale <= 0) { this->scale = 1.0; }

	this->red.proc.alpha = this->getTable()->GetSubTable("Red Filtering")->GetNumber("Alpha", this->red.proc.alpha);
	this->red.proc.beta = this->getTable()->GetSubTable("Red Filtering")->GetNumber("Beta", this->red.proc.beta);
	this->red.proc.gamma = this->getTable()->GetSubTable("Red Filtering")->GetNumber("Gamma", this->red.proc.gamma);
	this->red.proc.thresh = this->getTable()->GetSubTable("Red Filtering")->GetNumber("Thresh", this->red.proc.gamma);
	this->blue.proc.alpha = this->getTable()->GetSubTable("Blue Filtering")->GetNumber("Alpha", this->blue.proc.alpha);
	this->blue.proc.beta = this->getTable()->GetSubTable("Blue Filtering")->GetNumber("Beta", this->blue.proc.beta);
	this->blue.proc.gamma = this->getTable()->GetSubTable("Blue Filtering")->GetNumber("Gamma", this->blue.proc.gamma);
	this->blue.proc.thresh = this->getTable()->GetSubTable("Blue Filtering")->GetNumber("Thresh", this->blue.proc.gamma);

	const cv::Size2i fsz = io_frame.size() / this->scale;

	if(fsz != this->buffer.size()) {
		this->buffer = cv::Mat{fsz, CV_8UC3};
		for(size_t i = 0; i < 3; i++) {
			this->channels[i] = cv::Mat{fsz, CV_8UC1};
		}
		this->red.proc.binary = cv::Mat{fsz, CV_8UC1};
		this->blue.proc.binary = cv::Mat{fsz, CV_8UC1};
	}

	cv::resize(io_frame, this->buffer, fsz);
	cv::split(this->buffer, this->channels);

	cv::Mat annotations = cv::Mat::zeros(io_frame.size(), CV_8UC3);

	if(_red) {
		this->red.proc.threshold(this->channels);
		std::sort(this->red.proc.objects.begin(), this->red.proc.objects.end(), std::greater<>());	// sort results by size
		// temporal analysis here
		if(this->red.proc.objects.size() > 0) {
			this->red.targets.resize(this->red.proc.objects.size());
			cv::Mat1f tvec, rvec;
			std::array<cv::Point2f, 4> outline;
			for(size_t i = 0; i < this->red.proc.objects.size(); i++) {
				const CvCargo& v = this->red.proc.objects[i] *= this->scale;
				outline = {
					cv::Point2f(v.center.x - v.radius, v.center.y),
					cv::Point2f(v.center.x, v.center.y - v.radius),
					cv::Point2f(v.center.x + v.radius, v.center.y),
					cv::Point2f(v.center.x, v.center.y + v.radius)
				};
				cv::solvePnP(
					Cargo<>::world_coords, outline,
					this->getSrcMatrix(), this->getSrcDistort(),
					rvec, tvec
				);
				this->red.targets[i].update(reinterpret_cast<float*>(tvec.data));
				// if debug
				cv::circle(annotations, v.center, v.radius, ::markup_map[~this->red.B_CLR][0]);
			}
		}
		if(_contours) {
			::rescale(this->red.proc.contours, this->scale);
			cv::drawContours(annotations, this->red.proc.contours, -1, ::markup_map[~this->red.B_CLR][1]);
		}
	} else {
		for(RedCargo& target : this->red.targets) { target.setExpired(); }
	}
	if(_blue) {
		this->blue.proc.threshold(this->channels);
		std::sort(this->blue.proc.objects.begin(), this->blue.proc.objects.end(), std::greater<>());
		// temporal analysis here
		if(this->blue.proc.objects.size() > 0) {
			this->blue.targets.resize(this->blue.proc.objects.size());
			cv::Mat1f tvec, rvec;
			std::array<cv::Point2f, 4> outline;
			for(size_t i = 0; i < this->blue.proc.objects.size(); i++) {
				const CvCargo& v = this->blue.proc.objects[i] *= this->scale;
				outline = {
					cv::Point2f(v.center.x - v.radius, v.center.y),
					cv::Point2f(v.center.x, v.center.y - v.radius),
					cv::Point2f(v.center.x + v.radius, v.center.y),
					cv::Point2f(v.center.x, v.center.y + v.radius)
				};
				cv::solvePnP(
					Cargo<>::world_coords, outline, rvec, tvec,
					this->getSrcMatrix(), this->getSrcDistort()
				);
				this->blue.targets[i].update(reinterpret_cast<float*>(tvec.data));
				// if debug
				cv::circle(annotations, v.center, v.radius, ::markup_map[~this->blue.B_CLR][0]);
			}
		}
		if(_contours) {
			::rescale(this->blue.proc.contours, this->scale);
			cv::drawContours(annotations, this->blue.proc.contours, -1, ::markup_map[~this->blue.B_CLR][1]);
		}
	} else {
		for(BlueCargo& target : this->blue.targets) { target.setExpired(); }
	}

	if(_threshold && (_red || _blue)) {
		if(_red && _blue) {
			memcpy_bitwise_or_asm(this->red.proc.binary.data, this->blue.proc.binary.data, this->red.proc.binary.data, fsz.area());
			cv::cvtColor(this->red.proc.binary, this->buffer, cv::COLOR_GRAY2BGR);
		} else {
			cv::cvtColor((_blue ? this->blue.proc.binary : this->red.proc.binary), this->buffer, cv::COLOR_GRAY2BGR);
		}
		cv::resize(this->buffer, io_frame, io_frame.size(), cv::INTER_NEAREST);
	}
	cv::bitwise_or(annotations, io_frame, io_frame);

}