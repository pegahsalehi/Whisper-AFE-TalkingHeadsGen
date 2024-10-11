# Landmark Distance (LMD)
class LMDMeter:
    def __init__(self, backend='dlib', region='mouth'):
        self.backend = backend
        self.region = region # mouth or face

        if self.backend == 'dlib':
            import dlib

            # load checkpoint manually
            self.predictor_path = './shape_predictor_68_face_landmarks.dat'
            if not os.path.exists(self.predictor_path):
                raise FileNotFoundError('Please download dlib checkpoint from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')

            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(self.predictor_path)

        else:

            import face_alignment
            try:
                self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
            except:
                self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

        self.V = 0
        self.N = 0
    
    def get_landmarks(self, img):

        if self.backend == 'dlib':
            dets = self.detector(img, 1)
            for det in dets:
                shape = self.predictor(img, det)
                # ref: https://github.com/PyImageSearch/imutils/blob/c12f15391fcc945d0d644b85194b8c044a392e0a/imutils/face_utils/helpers.py
                lms = np.zeros((68, 2), dtype=np.int32)
                for i in range(0, 68):
                    lms[i, 0] = shape.part(i).x
                    lms[i, 1] = shape.part(i).y
                break

        else:
            lms = self.predictor.get_landmarks(img)[-1]
        
        # self.vis_landmarks(img, lms)
        lms = lms.astype(np.float32)

        return lms

    def vis_landmarks(self, img, lms):
        plt.imshow(img)
        plt.plot(lms[48:68, 0], lms[48:68, 1], marker='o', markersize=1, linestyle='-', lw=2)
        plt.show()

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            inp = inp.detach().cpu().numpy()
            inp = (inp * 255).astype(np.uint8)
            outputs.append(inp)
        return outputs
    
    def update(self, preds, truths):
        # assert B == 1
        preds, truths = self.prepare_inputs(preds[0], truths[0]) # [H, W, 3] numpy array

        # get lms
        lms_pred = self.get_landmarks(preds)
        lms_truth = self.get_landmarks(truths)

        if self.region == 'mouth':
            lms_pred = lms_pred[48:68]
            lms_truth = lms_truth[48:68]

        # avarage
        lms_pred = lms_pred - lms_pred.mean(0)
        lms_truth = lms_truth - lms_truth.mean(0)
        
        # distance
        dist = np.sqrt(((lms_pred - lms_truth) ** 2).sum(1)).mean(0)
        
        self.V += dist
        self.N += 1
    
    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LMD ({self.backend})"), self.measure(), global_step)

    def report(self):
        return f'LMD ({self.backend}) = {self.measure():.6f}' 
    


import os
import cv2
import numpy as np

# Initialize the LMDMeter with 'fan' backend
lmd_meter = LMDMeter(backend='fan')

# Set the paths for the predicted and ground truth image folders
pred_folder = r"/home/host/pegah/evl/RAD-NeRF/deep/obama/GD"
truth_folder = r"/home/host/pegah/evl/RAD-NeRF/deep/obama/GT"

# Get the sorted list of image filenames
pred_images = sorted(os.listdir(pred_folder))
truth_images = sorted(os.listdir(truth_folder))

# Iterate over the images and update the LMDMeter
for pred_img_name, truth_img_name in zip(pred_images, truth_images):
    # Load predicted and ground truth images
    pred_img_path = os.path.join(pred_folder, pred_img_name)
    truth_img_path = os.path.join(truth_folder, truth_img_name)

    pred_img = cv2.imread(pred_img_path)
    truth_img = cv2.imread(truth_img_path)

    # Convert images from BGR to RGB
    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
    truth_img = cv2.cvtColor(truth_img, cv2.COLOR_BGR2RGB)

    # Ensure images are in uint8 format
    pred_img = pred_img.astype(np.uint8)
    truth_img = truth_img.astype(np.uint8)

    # Directly get landmarks without using prepare_inputs
    lms_pred = lmd_meter.get_landmarks(pred_img)
    lms_truth = lmd_meter.get_landmarks(truth_img)

    # Adjust the landmarks based on the specified region
    if lmd_meter.region == 'mouth':
        lms_pred = lms_pred[48:68]
        lms_truth = lms_truth[48:68]

    # Calculate the distance
    lms_pred = lms_pred - lms_pred.mean(0)
    lms_truth = lms_truth - lms_truth.mean(0)
    
    dist = np.sqrt(((lms_pred - lms_truth) ** 2).sum(1)).mean(0)

    # Update the cumulative distance and count
    lmd_meter.V += dist
    lmd_meter.N += 1

# Calculate and print the final LMD
final_lmd = lmd_meter.measure()
print(f'Final Landmark Distance (LMD): {final_lmd:.6f}')

