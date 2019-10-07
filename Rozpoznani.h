#pragma once

#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include "Detekce.h" 
#include "PreProcess.h"
#include <dirent.h>
#include <string.h>
#include <sys/stat.h>
#include <map>

#define LABEL_DIR "labels/"

inline bool exists_test(const std::string& name);

std::vector<int> Getlabels();

std::string GetName();

void read_names();

std::map<int, std::string> GetNames();

void Clear();

//! Nacteni databaze
std::vector<cv::Mat> get_train_database(cv::Mat processed_test_image, int index, cv::Rect is_face, int *num_of_images);

//! Natrenovani modelu
cv::Ptr<cv::face::FaceRecognizer> GetModel(const std::vector<cv::Mat> train_images, const std::vector<int> labels);

//! Rozpoznani obliceje
bool Recognition(const cv::Ptr<cv::face::FaceRecognizer> model, cv::Mat processed_test_image);

//! Vytvori priblizne zrekonstruovanou tvar projekcemi vlastnich vektoru a vlastnich hodnot daneho (predem zpracovaneho) obliceje. Pouziva se pro FisherFace a EigenFace.
/*!
	cv::Mat get_reconstruct(const cv::Ptr<cv::face::FaceRecognizer> model, const cv::Mat processed_test_image); 
*/

//! Porovnava dve fotky mezi sebou pomoci Euklidovske normy.
double get_distance(const cv::Mat x, const cv::Mat y);