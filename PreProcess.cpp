const double leye_position_x = 0.18;  /*!< pozadovana hodnota */
const double leye_position_y = 0.15;
const double ellipse_y = 0.38;	
const double ellipse_w = 0.55;         
const double ellipse_h = 0.82;         

#include "Detekce.h"       
#include "PreProcess.h"    
							
using namespace cv;
using namespace std;

void side_equalization(Mat &input)
{
	int input_w = input.cols;
	int input_h = input.rows;

	Mat face_eq;
	equalizeHist(input, face_eq);

	int median = input_w / 2;
	Mat left_side_eq = input(Rect(0, 0, median, input_h));
	Mat rigth_side_eq = input(Rect(median, 0, input_w - median, input_h));
	equalizeHist(left_side_eq, left_side_eq);
	equalizeHist(rigth_side_eq, rigth_side_eq);

	for (int y = 0; y<input_h; y++) {
		for (int x = 0; x<input_w; x++) {
			int outputPixel;
			if (x < input_w / 4) {      
				outputPixel = left_side_eq.at<uchar>(y, x);
			}
			else if (x < input_w * 2 / 4) {
				int left_side_pixel = left_side_eq.at<uchar>(y, x);
				int face_eq_pixel = face_eq.at<uchar>(y, x);
				float f = (x - input_w * 1 / 4) / (float)(input_w*0.25f);
				outputPixel = cvRound((1.0f - f) * left_side_pixel + (f)* face_eq_pixel);
			}
			else if (x < input_w * 3 / 4) {  
				int right_side_pixel = rigth_side_eq.at<uchar>(y, x - median);
				int face_eq_pixel = face_eq.at<uchar>(y, x);
				float f = (x - input_w * 2 / 4) / (float)(input_w*0.25f);
				outputPixel = cvRound((1.0f - f) * face_eq_pixel + (f)* right_side_pixel);
			}
			else {            
				outputPixel = rigth_side_eq.at<uchar>(y, x - median);
			}
			input.at<uchar>(y, x) = outputPixel;
		}
	}
}

void eyes_detection(const Mat &face_image, CascadeClassifier &haar_eye, CascadeClassifier &haar_eye_glasses, Point &left_eye_center, Point &right_eye_center)
{
	const float current_x_position = 0.15f;
	const float current_y_position = 0.24f;
	const float current_w = 0.32f;
	const float current_h = 0.29f;

	int left_x_eye_area = cvRound(face_image.cols * current_x_position);
	int top_y_eye_area = cvRound(face_image.rows * current_y_position);
	int width_x_eye_area = cvRound(face_image.cols * current_w);
	int height_y_eye_area = cvRound(face_image.rows * current_h);
	int right_x_eye_area = cvRound(face_image.cols * (1.0 - current_x_position - current_w));
	Mat left_eye_area = face_image(Rect(left_x_eye_area, top_y_eye_area, width_x_eye_area, height_y_eye_area));
	Mat right_eye_area = face_image(Rect(right_x_eye_area, top_y_eye_area, width_x_eye_area, height_y_eye_area));
	Rect is_left_eye, is_right_eye;

	face_detection(left_eye_area, haar_eye, is_left_eye, left_eye_area.cols);
	face_detection(right_eye_area, haar_eye, is_right_eye, right_eye_area.cols);
	if (is_left_eye.width <= 0 && !haar_eye_glasses.empty())
		face_detection(left_eye_area, haar_eye_glasses, is_left_eye, left_eye_area.cols);

	if (is_right_eye.width <= 0 && !haar_eye_glasses.empty())
		face_detection(right_eye_area, haar_eye_glasses, is_right_eye, right_eye_area.cols);

	if (is_left_eye.width > 0) {
		is_left_eye.x += left_x_eye_area;
		is_left_eye.y += top_y_eye_area;
		left_eye_center = Point(is_left_eye.x + is_left_eye.width / 2, is_left_eye.y + is_left_eye.height / 2);
	}
	else {
		left_eye_center = Point(-1, -1);
	}

	if (is_right_eye.width > 0) {
		is_right_eye.x += right_x_eye_area;
		is_right_eye.y += top_y_eye_area;
		right_eye_center = Point(is_right_eye.x + is_right_eye.width / 2, is_right_eye.y + is_right_eye.height / 2);
	}
	else {
		right_eye_center = Point(-1, -1);
	}
}

Mat process_img(Mat &src_img, int face_width, CascadeClassifier &haar_face, CascadeClassifier &haar_eye, CascadeClassifier &haar_eye_glasses, bool diff_light, Rect *is_face, Point *left_eye, Point *right_eye, Rect *is_leye, Rect *is_reye)
{
	Rect face_area;
	int face_height = face_width;

	face_detection(src_img, haar_face, face_area);

	if (is_face)
		is_face->width = -1;
	if (left_eye)
		left_eye->x = -1;
	if (right_eye)
		right_eye->x = -1;
	if (is_leye)
		is_leye->width = -1;
	if (is_reye)
		is_reye->width = -1;

	if (face_area.width > 0) {

		if (is_face)
			*is_face = face_area;

		Mat input = src_img(face_area);  
		
		/*namedWindow("input");
		imshow("input", input);*/

		Mat grayscale;
		if (input.channels() == 3) {
			cvtColor(input, grayscale, CV_BGR2GRAY);
		}
		else {
			grayscale = input;
		}

		Point left_eye_center, right_eye_center;
		eyes_detection(grayscale, haar_eye, haar_eye_glasses, left_eye_center, right_eye_center);

		if (left_eye)
			*left_eye = left_eye_center;
		if (right_eye)
			*right_eye = right_eye_center;

		if (left_eye_center.x >= 0 && right_eye_center.x >= 0) {

			Point2f eyes_center = Point2f((left_eye_center.x + right_eye_center.x) * 0.5f, (left_eye_center.y + right_eye_center.y) * 0.5f);
			double dy = (right_eye_center.y - left_eye_center.y);
			double dx = (right_eye_center.x - left_eye_center.x);
			double current_length = sqrt(dx*dx + dy*dy);
			double angle = atan2(dy, dx) * 180.0 / CV_PI; 

			const double reye_position_x = (1.0f - leye_position_x);
			double length = (reye_position_x - leye_position_x) * face_width;
			double scale = length / current_length;
		
			Mat rotation_matrix = getRotationMatrix2D(eyes_center, angle, scale);
			rotation_matrix.at<double>(0, 2) += face_width * 0.5f - eyes_center.x;
			rotation_matrix.at<double>(1, 2) += face_height * leye_position_y - eyes_center.y;

			Mat affine_trans_img = Mat(face_height, face_width, CV_8U, Scalar(128));
			warpAffine(grayscale, affine_trans_img, rotation_matrix, affine_trans_img.size());

			if (!diff_light) {
				equalizeHist(affine_trans_img, affine_trans_img);
			}
			else {
				side_equalization(affine_trans_img);
			}

			Mat filtered = Mat(affine_trans_img.size(), CV_8U);
			bilateralFilter(affine_trans_img, filtered, 0, 20.0, 2.0);

			Mat mask = Mat(affine_trans_img.size(), CV_8U, Scalar(0)); 
			Point center = Point(face_width / 2, cvRound(face_height * ellipse_y));
			Size size = Size(cvRound(face_width * ellipse_w), cvRound(face_height * ellipse_h));
			ellipse(mask, center, size, 0, 0, 360, Scalar(255), CV_FILLED);

			Mat result = Mat(affine_trans_img.size(), CV_8U, Scalar(128)); 
			filtered.copyTo(result, mask); 

			return result;
		}
		
	}
	else { cout << "Nepodarilo se detekovat oblicej" << endl; 
	cout << endl;
	}
	return Mat();
}