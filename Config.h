#pragma once

//! Nacteme potrebne soubory pro detekci obliceje a oci
const char *haar_face = "C:/haarcascade_frontalface_alt.xml"; 
const char *haar_eye = "C:/haarcascade_eye.xml";             
const char *haar_eye_glasses = "C:/haarcascade_eye_tree_eyeglasses.xml";

//! Cesty k potrebnym souborum
#define MODEL_DIR "model/"
#define LABEL_DIR "labels/"
#define TRAINING_DIR "training/"

// Escape
#if !defined VK_ESCAPE
#define VK_ESCAPE 0x1B
#endif


