#pragma once

#include <opencv2/opencv.hpp>
#include <core/calib.h>


static const inline CalibList
	calibrations{
		{
			{ "lifecam_hd3000", {
				{ cv::Size{640, 480}, {
					cv::Mat1f{{3, 3},{
						673.6653136395231, 0, 339.861572657799,
						0, 666.1104961259615, 244.21065776461745,
						0, 0, 1}},
					cv::Mat1f{{1, 5},{
						0.04009256446529976, -0.4529245799337021,
						-0.001655316303789686, -0.00019284071985319236,
						0.5736326357832554}}
				} }
			} },
			{ "logitech_b910", {
				{ cv::Size{640, 480}, {
					cv::Mat1f{
						561.4074313049189, 0, 337.76418985338495,
						0, 557.6888346113145, 237.32558163183825,
						0, 0, 1},
					cv::Mat1f{
						0.03335218620375778, -0.42937056677801017,
						-0.001215946872533738, 0.011464765688132103,
						1.0705120077832195}
				} }
			} }
		}
	}
;