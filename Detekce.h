#pragma once

#include <stdio.h>
#include <iostream>
#include <vector>

#include "opencv2/opencv.hpp"
#include "opencv2/face.hpp"

//! Detekce obliceje. Detecke nejvetsiho objektu.
/*!
	CASCADE_FIND_BIGGEST_OBJECT je hodnota, ktera se pouziva pro detekci 
	nejvetsiho objektu.
*/
void face_detection(const cv::Mat &img, cv::CascadeClassifier &cascade, cv::Rect &object, int new_width = 300);

void detect(const cv::Mat &input_image, cv::CascadeClassifier &cascade, std::vector<cv::Rect> &objects, int new_width, int flags, cv::Size min_size, float scale_factor, int min_neighbors);
