#include "Rozpoznani.h"  
using namespace cv;
using namespace cv::face;
using namespace std;

//! Prahova hodnota
/*!
Nastaveni prahove hodnoty, pokud chceme pouzit vlastne implementovanou funkci
porovnani vstupniho testovaciho obrazku a trenovaci mnoziny.
*/
const float threshold = 0.45f;

//! Jmena uzivatelu
vector<int> labels;

map<int, string> Map_names;
vector<string> directories;

vector<Mat> train_images;
Mat old_train_images;

//! Identifikace osoby
int person = -1;

//! Pocet lidi
int num_of_Persons = 0;

int id = -1;

// Parametry, ktere ukazuji o kolik se museji snimky lisit mezi sebou. 
const double threshold_distance = 0.3;      
const double threshold_time_interval = 1.0; 

void read_names() {
	if (exists_test(string(LABEL_DIR) + "Names.txt"))
	{
		int i = 1;
		cout << "Nacitani jmen..." << endl;
		ifstream inFile(string(LABEL_DIR) + "Names.txt");
		string curr_name;
		while (inFile >> curr_name) {
			Map_names[i] = curr_name;
			i++;
		}
		inFile.close();
	}
}

vector<int> Getlabels() {
	return labels;
}

map<int, string> GetNames() {
	return Map_names;
}

void Clear() {
	labels.clear();
	Map_names.clear();
	train_images.clear();
}

inline bool exists_test(const string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

vector<Mat> get_train_database(Mat processed_test_image, int index, Rect is_face, int *num_of_images) {
	double old_time = 0;

	bool is_founded = false;
	if (processed_test_image.data)
		is_founded = true;

	if (is_face.width > 0 && is_founded) {
		/*! Overime si, jestli oblicej vypada odlisne od predhoziho zpracovaneho obliceje.*/
		double current_distance = 9000000000.0;
		if (old_train_images.data) {
			current_distance = get_distance(processed_test_image, old_train_images);
		}

		double current_time = (double)getTickCount();
		double time_interval = (current_time - old_time) / getTickFrequency();

		/*! Ulozime oblicej pokud on se lesi od predchoziho a prosel dostatecny cas.*/
		if ((current_distance > threshold_distance) && (time_interval > threshold_time_interval)) {
			train_images.push_back(processed_test_image);
			labels.push_back(index);

			num_of_images[0]++;
			
			cout << "Ulozime oblicej " << train_images.size() << " pro osobu " << index << endl;
			cout <<endl;

			old_train_images = processed_test_image;
			old_time = current_time;
			
			/*! Spustime pocitadlo aby vsechne procesy stihly probehnout.*/
			Sleep(1000);
		}
		else {
			cout << "Obrazek je podobny predchozimu " << endl;
			cout << endl;
		}
	}

	cout << "Hotovo." << endl;

	return train_images;
}

Ptr<FaceRecognizer> GetModel(const vector<Mat> train_images, const vector<int> labels)
{
	Ptr<FaceRecognizer> model;
	
	model = LBPHFaceRecognizer::create(1, 8, 4, 4, 10);
	
	if (model.empty()) {
		cerr << "ERROR: LBPH algoritm nefunguje s danou verzi OpenCV." << endl;
		exit(1);
	}

	model->train(train_images, labels);

	cout << "Model je uspesne natrenovan" << endl;
	return model;
}

string GetName() {
	return Map_names[id];
}

double get_distance(const Mat x, const Mat y)
{
	if (x.rows > 0 && x.rows == y.rows && x.cols > 0 && x.cols == y.cols) {
		double Euclidean_norm = norm(x, y, CV_L2);
		double confidence = Euclidean_norm / (double)(x.rows * x.cols);
		return confidence;
	}
	else {
		return 9000000000.0;
	}
}

bool Recognition(const Ptr<FaceRecognizer> model,   Mat processed_test_image)
{
	string outputStr;
	
	if (Map_names.empty()) {
		read_names();
	}

	//! Threshold, ktery dostavame presne z algoritmu
	double confidence = 0.0;
	int label = -1;
	model->predict(processed_test_image, label, confidence);
	if (label > -1 && confidence < 10) {
		id = label;
		outputStr = Map_names[label];
		cout << "Identita: " << outputStr << ". Podobnost: " << confidence << endl;
		return true;
	}
	else
	{
		outputStr = "Unknown";
		cout << "Identita: " << outputStr << ". Podobnost: " << confidence << endl;
		return false;
	}
	/*! Vlastni vypocet distance */
		//int identity = -1;
		//Mat reconstruction;
		//reconstruction = get_reconstruct(model, processed_test_image);
		//double confidence = get_distance(processed_test_image, reconstruction);
		//if (confidence < threshold) {
		//	identity = model->predict(processed_test_image);
		//	id = identity;
		//	outputStr = Map_names[identity];
		//	cout << "Identita: " << outputStr << ". Podobnost: " << confidence << endl;
		//	return true;
		//}
		//else {
		//	outputStr = "Unknown";
		//	cout << "Identita: " << outputStr << ". Podobnost: " << confidence << endl;
		//	return false;
		//}

	}

/*! Vlastni funkce pro vzpocet podobnosti */
//Mat get_reconstruct(const Ptr<FaceRecognizer> model, const Mat processed_test_image)
//{
//	try {
//
//		Mat evs = model->getEigenVectors(); 
//		Mat mean = model->getMean(); 
//
//		/*! Projekce vstupniho obrazku na PCA podprostor.*/
//		Mat projection = LDA::subspaceProject(evs, mean, processed_test_image.reshape(1, 1));
//
//		/*! Rekonstrukce obliceje zpatky z PCA podprostoru.*/
//		Mat reconstruction_from_pca = LDA::subspaceReconstruct(evs, mean, projection);
//
//		Mat reconstruction_mat = reconstruction_from_pca.reshape(1, processed_test_image.rows);
//		Mat reconstruction = Mat(reconstruction_mat.size(), CV_8U);
//		reconstruction_mat.convertTo(reconstruction, CV_8U, 1, 0);
//
//		return reconstruction;
//
//	}
//	catch (Exception e) {
//		return Mat();
//	}
//}

