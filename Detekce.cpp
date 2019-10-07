#include "Detekce.h"


using namespace cv;
using namespace std;

//! Detekce obliceje.
void face_detection(const Mat &input_image, CascadeClassifier &cascade, Rect &object, int new_width)
{
	int flags = CASCADE_FIND_BIGGEST_OBJECT;
	Size min_size = Size(20, 20);
	float scale_factor = 1.1f;
	int min_neighbors = 4;

	vector<Rect> objects;
	detect(input_image, cascade, objects, new_width, flags, min_size, scale_factor, min_neighbors);
	if (objects.size() > 0) {
		object = (Rect)objects.at(0);
	}
	else {
		object = Rect(-1, -1, -1, -1);
	}
}

void detect(const Mat &input_image, CascadeClassifier &cascade, vector<Rect> &objects, int new_width, int flags, Size min_size, float scale_factor, int min_neighbors)
{
	Mat grayscale;
	if (input_image.channels() == 3) {
		cvtColor(input_image, grayscale, CV_BGR2GRAY);
	}
	else {
		grayscale = input_image;
	}

	Mat resized_image;
	float scale = input_image.cols / (float)new_width;
	if (input_image.cols > new_width) {
		int new_height = cvRound(input_image.rows / scale);
		resize(grayscale, resized_image, Size(new_width, new_height));
	}
	else {
		resized_image = grayscale;
	}

	Mat equalized_image;
	equalizeHist(resized_image, equalized_image);

	cascade.detectMultiScale(equalized_image, objects, scale_factor, min_neighbors, flags, min_size);

	if (input_image.cols > new_width) {
		for (int i = 0; i < (int)objects.size(); i++) {
			objects[i].x = cvRound(objects[i].x * scale);
			objects[i].y = cvRound(objects[i].y * scale);
			objects[i].height = cvRound(objects[i].height * scale);
			objects[i].width = cvRound(objects[i].width * scale);
		}
	}

}

