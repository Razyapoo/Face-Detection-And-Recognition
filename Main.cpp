
//! Velikosti obliceje pro rychlejsi zpracovani
const int width = 90;
const int height = width;

const int thickness = 6;

/*! 
	Zpracovani prave a leve strany obliceje zvlast ve pripade,
	ze maji ruznou intenzitu svetla.
*/
const bool diff_light = true; 

#include "Detekce.h"       
#include "PreProcess.h"    
#include "Rozpoznani.h"
#include "config.h"

using namespace cv;
using namespace std;
using namespace cv::face;

//! Nastaveni pro ruzne rezimy programu
enum Mode {
	start = 0,
	add_person,
	detection,
	collection,
	training,
	recognition,
	delete_all,
	end
};

Mode mode = detection;
bool rec = false;
int index = 0;
int num_of_images = 0;
bool mode_r = false;

Rect add_button;
Rect rec_button;
Rect det_button;
Rect train_button;
Rect del_button;

Ptr<FaceRecognizer> model;
map<int, string> names;

//! Nacteni potrebnych kaskadovych klasifikatoru
void open_cascade_files(CascadeClassifier &face_cascade_file, CascadeClassifier &eye_cascade_file, CascadeClassifier &eye_glasses_cascade_file)
{
	try {   
		face_cascade_file.load(haar_face);
	}
	catch (Exception &e) {}
	if (face_cascade_file.empty()) {
		cout << "ERROR" << endl;
		exit(1);
	}
	cout << "Soubor " << haar_face << " je uspesne otevren." << endl;
	
	try {   
		eye_cascade_file.load(haar_eye);
	}
	catch (Exception &e) {}
	if (eye_cascade_file.empty()) {
		cout << "ERROR" << endl;
		exit(1);
	}
	cout << "Soubor " << haar_eye << " je uspesne otevren." << endl;

	try {  
		eye_glasses_cascade_file.load(haar_eye_glasses);
	}
	catch (Exception &e) {}
	if (eye_glasses_cascade_file.empty()) {
		cout << "ERROR" << endl;
	}
	else
		cout << "Soubor " << haar_eye_glasses << " je uspesne otevren." << endl;
}

//! Ziskani pristupu k webove kamere pocitace
void open_webcam(VideoCapture &vc, int cnum) {
	try {
		vc.open(cnum); 
	}
	catch (cv::Exception &e) {}
	if (!vc.isOpened()) {
		cerr << "ERROR" << endl;
		exit(1);
	}
	cout << "Webova kamera " << cnum << " je uspesne otevrena." << endl;
}

//! Kresleni textu uvnitr obdelniku
Rect put_text(Mat test_image, string text, Point coordinate, Scalar color)
{
	int line = 0;
	Size size_of_text = getTextSize(text, FONT_HERSHEY_DUPLEX, 0.8f, 1, &line);
	line += 1;

	if (coordinate.x < 0) {
		coordinate.x += test_image.cols - size_of_text.width + 1;
	}

	if (coordinate.y >= 0) {
		coordinate.y += size_of_text.height;
	}
	else {
		coordinate.y += test_image.rows - line + 1;
	}

	Rect rect = Rect(coordinate.x, coordinate.y - size_of_text.height, size_of_text.width, line + size_of_text.height);

	putText(test_image, text, coordinate, FONT_HERSHEY_DUPLEX, 0.8f, color, 1, CV_AA);

	return rect;
}

//! Kresleni tlacitek
Rect create_button(Mat test_image, string text, Point coordinate, int min = 0) {
	Point text_coordinates = Point(coordinate.x + thickness, coordinate.y + thickness);
	Rect text_in_rect = put_text(test_image, text, text_coordinates, CV_RGB(0, 0, 0));
	Rect control_button = Rect(text_in_rect.x - thickness, text_in_rect.y - thickness, text_in_rect.width +  thickness, text_in_rect.height + thickness);
	if (control_button.width < min)
		control_button.width = min;

	Mat matBtn = test_image(control_button);
	matBtn += CV_RGB(121, 121, 121);
	rectangle(test_image, control_button, CV_RGB(255, 255, 255), 1, CV_AA);

	put_text(test_image, text, text_coordinates, CV_RGB(26, 26, 26));

	return control_button;
}

bool is_click(const Point point, const Rect rect)
{
	if (point.x >= rect.x && point.x <= (rect.x + rect.width - 1))
		if (point.y >= rect.y && point.y <= (rect.y + rect.height - 1))
			return true;

	return false;
}

//! Manipulace s mysy
void onMouse(int event, int x, int y, int flags, void* userdata)
{
	/*! Pozivame jen leve tlacitko mysy*/ 
	if (event != CV_EVENT_LBUTTONDOWN)
		return;

	Point point = Point(x, y);
	if (is_click(point, add_button)) {
		cout << "Pridani dalsiho uzivatele..." << endl;
		mode = add_person;
	}
	else if (is_click(point, rec_button)) {
		if (exists_test(string(MODEL_DIR) + "model.xml")) {
			mode = training;
			mode_r = true;
		}	
		else if (mode == training){
			cout << "Uzivatel chce zacit trenovat." << endl;
			mode = recognition;
		}
		else {
			cout << "Nepodarilo se najit model..." << endl;
		}
	}
	else if (is_click(point, det_button)) {
		mode = detection;
	}
	else if (is_click(point, train_button)) {
		if (exists_test(string(MODEL_DIR) + "model.xml") || mode == collection ){
			mode = training;
		}
		else {
			cout << "Nepodarilo se najit model..." << endl;
		}
	}
	else if (is_click(point, del_button)) {
		if (exists_test(string(MODEL_DIR) + "model.xml")) {
			mode = delete_all;
		}
		else {
			cout << "Model neexistuje..." << endl;
		}
	}

}


//! Natrenovani modelu a rozpoznani osoby
void rec_image(VideoCapture &vc, CascadeClassifier &face_cascade_file, CascadeClassifier &eye_cascade_file, CascadeClassifier &eye_glasses_cascade_file)
{
	
	vector<Mat> train_images;
	vector<int> labels;
	int sizeText = 0;
	string name;
	stringstream ss;
	bool ifDelete = false;

	while (true)
	{
		Mat test_image;
		vc.read(test_image);

		Rect is_face; 
		Rect is_leye, is_reye; 
		Point left_eye, right_eye; 
		//! Ziskani zpracovaneho testovaciho obrazku
		Mat processed_test_image = process_img(test_image, width, face_cascade_file, eye_cascade_file, eye_glasses_cascade_file, diff_light, &is_face, &left_eye, &right_eye, &is_leye, &is_reye);

		int _baseline = 0;
		_baseline += 3;

		// center the text
		Point _textOrg(thickness + 10, (thickness + 450));
		Point _textOrg1(thickness + 12, (thickness + 470));

		bool is_founded = false;
		if (processed_test_image.data)
			is_founded = true;


		if (mode == detection) {
			rec = false;
			putText(test_image, "Detekce obliceju", _textOrg, CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(255), 1.5, 8);
		}
		else if (mode == add_person) {
			if (is_founded) {
				if (ifDelete) {
					Clear();
					ifDelete = false;
					index = 0;
				}
				index++;
				if (!exists_test(string(MODEL_DIR) + "model.xml")) {
					putText(test_image, "Pridani dalsiho uzivatele", _textOrg, CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(255), 1.5, 8);
					STARTUPINFO info = { sizeof(info) };
					PROCESS_INFORMATION processInfo;
					if (CreateProcess(TEXT("JmenoUzivatele.exe"), NULL, NULL, NULL, TRUE, 0, NULL, NULL, &info, &processInfo))
					{
						WaitForSingleObject(processInfo.hProcess, INFINITE);
						CloseHandle(processInfo.hProcess);
						CloseHandle(processInfo.hThread);
					}
					read_names();
					names = GetNames();
					if (names.empty())
						mode = detection;
					else
						mode = collection;
					names.clear();
					num_of_images = 0;
				}
				else {
					putText(test_image, "Model uz existuje, pro smazani modelu zmacknete Delete...", _textOrg, CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(255), 1.5, 8);
				}
			}

		}
		else if (mode == collection) {
			putText(test_image, "Start Collection", _textOrg, CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(255), 1.5, 8);

			if (num_of_images < 30)
			{
				train_images = get_train_database(processed_test_image, index, is_face, &num_of_images);
			}
			else {
				putText(test_image, "Dosahnut maximalni pocet obrazku pro jedneho cloveka", _textOrg1, CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(255), 1.5, 8);
			}
		}
		else if (mode == training) {

			if (exists_test(string(MODEL_DIR) + "model.xml")) 
			{
				if (model == NULL) {
					putText(test_image, "Nacitani modelu...", _textOrg, CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(255), 1.5, 8);
					model = LBPHFaceRecognizer::create(1, 8, 4, 4, 10);
					model->read(string(MODEL_DIR) + "model.xml");
				}
				
				if (exists_test(string(LABEL_DIR) + "labels.txt") && labels.size() == 0)
				{
						putText(test_image, "Nacitani labelu...", _textOrg, CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(255), 1.5, 8);
						ifstream inFile(string(LABEL_DIR) + "labels.txt");
						int curr_num = 0;
						while (inFile >> curr_num) {
							labels.push_back(curr_num);
							labels.push_back(curr_num);
						}
						inFile.close();
					
				}
				if (exists_test(string(LABEL_DIR) + "Names.txt") && names.size() == 0) {
					read_names();
				}
			}
			else {
				labels = Getlabels();

				putText(test_image, "Vytvareni modelu...", _textOrg, CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(255), 1.5, 8);
				model = GetModel(train_images, labels);

				names = GetNames();

				string filepath = string(MODEL_DIR) + "model.xml";
				ss << "Ukladani modelu do " << filepath << "..." << flush;
				putText(test_image, ss.str(), _textOrg, CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(255), 1.5, 8);
				model->save(filepath);

				filepath = string(LABEL_DIR) + "labels.txt";
				ofstream datal(filepath);
				ss << "Ukladani labelu do " << filepath << "..." << flush;
				putText(test_image, ss.str(), _textOrg, CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(255), 1.5, 8);
				for (auto &&i : labels)
					datal << to_string(i) << endl;
				datal.close();
				putText(test_image, "[DONE]", _textOrg, CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(255), 1.5, 8);

			}
			mode = start;
			if (mode_r) {
				mode = recognition; 
				mode_r = false;
			}
		}
		else if (mode == recognition) {

			if (is_founded)
				rec = Recognition(model, processed_test_image);
			else
				rec = false;

			name = GetName();
		}
		else if (mode == delete_all) {
			train_images.clear();
			labels.clear();
			rec = false;
			names.clear();
			model.release();
			processed_test_image.release();
			if (exists_test(string(MODEL_DIR) + "model.xml")) {
				putText(test_image, "Delete model...", _textOrg, CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(255), 1.5, 8);
				remove("\model\\model.xml");
			}

			if (exists_test(string(LABEL_DIR) + "Names.txt")) {
				putText(test_image, "Delete Names...", _textOrg, CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(255), 1.5, 8);
				remove("\labels\\Names.txt");
			}

			if (exists_test(string(LABEL_DIR) + "labels.txt")) {
				putText(test_image, "Delete labels...", _textOrg, CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(255), 1.5, 8);
				remove("\labels\\labels.txt");
			}
			ifDelete = true;
			mode = detection;

		}

		int line = 0;
		Size size_of_text = getTextSize(name, CV_FONT_HERSHEY_SIMPLEX, 0.5, 1.5, &line);
		line += 3;

		Point textOrg(is_face.x, (is_face.y + is_face.height + 15));

		//! Kresleni obdelniku
		/*! 
			Pokud je tvar rozpoznana, nakresli se kolem ni zeleny obdelnik,
			jinak cerveny.
		*/
		if (is_face.width > 0 && rec) {
			rectangle(test_image, is_face, CV_RGB(0, 255, 0), 2, CV_AA);

			putText(test_image, name, textOrg, CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(255), 1.5, 8);

			//! Modre krouhy kolem oci
			Scalar eyeColor = CV_RGB(0, 255, 255);
			if (left_eye.x >= 0) {   
				circle(test_image, Point(is_face.x + left_eye.x, is_face.y + left_eye.y), 6, eyeColor, 1, CV_AA);
			}
			if (right_eye.x >= 0) { 
				circle(test_image, Point(is_face.x + right_eye.x, is_face.y + right_eye.y), 6, eyeColor, 1, CV_AA);
			}
			putText(test_image, "Dvere je otevrena", _textOrg, CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(255), 1.5, 8);
		}
		else {
			rectangle(test_image, is_face, CV_RGB(255, 0, 0), 2, CV_AA);

			putText(test_image, "Unknown", textOrg, CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar::all(255), 1.5, 8);

			Scalar eyeColor = CV_RGB(0, 255, 255);
			if (left_eye.x >= 0) { 
				circle(test_image, Point(is_face.x + left_eye.x, is_face.y + left_eye.y), 6, eyeColor, 1, CV_AA);
			}
			if (right_eye.x >= 0) { 
				circle(test_image, Point(is_face.x + right_eye.x, is_face.y + right_eye.y), 6, eyeColor, 1, CV_AA);
			}
				/*namedWindow("test_image");
				imshow("test_image", test_image);*/
		}

		//! Vytvoreni tlacitek
		add_button = create_button(test_image, "Add New Person", Point(thickness, thickness));
		det_button = create_button(test_image, "Detection", Point(add_button.x, add_button.y + add_button.height), add_button.width);
		rec_button = create_button(test_image, "Recognition", Point(det_button.x, det_button.y + det_button.height), det_button.width);
		train_button = create_button(test_image, "Training", Point(rec_button.x, rec_button.y + rec_button.height), rec_button.width);
		del_button = create_button(test_image, "Delete", Point(train_button.x, train_button.y + train_button.height), train_button.width);

		//! Vystupni okenko
		imshow("Face Recognition", test_image);

		char keypress = waitKey(20);  
		if (keypress == VK_ESCAPE) {  
			cout << "" << endl;
			break;
		}

	}
}

int main(int argc, char *argv[]) {
	CascadeClassifier face_cascade_file;
	CascadeClassifier eye_cascade_file;
	CascadeClassifier eye_glasses_cascade_file;
	vector<Rect> objects;
	VideoCapture vc;

	open_cascade_files(face_cascade_file, eye_cascade_file, eye_glasses_cascade_file);

	int cnum = 0; 
	if (argc > 1) {
		cnum = atoi(argv[1]);
	}

	//! Pripojeni pocitacove kamery
	open_webcam(vc, cnum);

	namedWindow("Face Recognition");
	setMouseCallback("Face Recognition", onMouse, NULL);

	//! Rozpoznani
	rec_image(vc, face_cascade_file, eye_cascade_file, eye_glasses_cascade_file);

	waitKey(0);

	return 0;
}
