#pragma once


#include <stdio.h>
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"

//! Vyhledavani oci. 
/*! 
	Metoda vrati stredy oci, pokud jsou detekovany, jinak vrati dvojici (-1, -1) 
*/
void eyes_detection(const cv::Mat &face, cv::CascadeClassifier &haar_eye, cv::CascadeClassifier &haar_eye_glasses, cv::Point &left_eye_center, cv::Point &right_eye_center);

//! Normalizace jasu, ve pripade, ze nejaka cast obrazku ma odlisnou intenzitu nez jina.
void side_equalization(cv::Mat &input);

//! Zpracovani obrazu pro lepsi vzsledek rozpoznani.
/*!
	Metoda v sobe zahrnuje zakladni upravu obrazku (napr. rotace, zmena meritka, vzhlazovani, 
	vyriznuti jen potrebnecasti obrazku atd.) Vyrovnani obrazku je provedeno na zaklade pozici oci.
*/
cv::Mat process_img(cv::Mat &srcImg, int face_width, cv::CascadeClassifier &haar_face, cv::CascadeClassifier &haar_eye, cv::CascadeClassifier &haar_eye_glasses, bool diff_light, cv::Rect *is_face = NULL, cv::Point *left_eye = NULL, cv::Point *right_eye = NULL, cv::Rect *is_leye = NULL, cv::Rect *is_reye = NULL);

