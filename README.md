## Multi-modal online interaction EMPathy (Multi-EMP) Dataset

![Dataset Environment](fig1.png)

- **Participants**: 66 participants forming 33 pairs (male-male and female-female combinations)
- **Interaction Format**: Unscripted online video conversations with keyword prompts
- **Session Structure**: 
  - Four 10-minute conversation sessions (2-min preparation + 8-min conversation)
  - Participants alternating between talker and listener roles
  - Topics covering both positive and negative experiences
 
The dataset encompasses four primary modalities and features extracting code is provided:
1. **Visual**: 
 - Facial features (FaceNet512, Facial Emotion Recognition)
 - Pose (head nods, gestures, movements)
 - Gaze direction and pupil positions

2. **Audio**: 
   - HuBERT features
   - Wav2Vec2.0 features

3. **Text**: 
   - Transcribed conversation content
   - DistilKoBERT embeddings

4. **Bio-signals**:
   - Electrodermal Activity (EDA)
   - Blood Volume Pulse (BVP)
   - Temperature (TEMP)
   - Metabolic Equivalent of Task (MET)
  
## Dataset Access
To request access to the Multi-EMP dataset, please send an email to **218354@jnu.ac.kr** with the following:

Subject: "Multi-EMP Dataset Access Request"
Attach **EULA-Multi-EMP.pdf** describing your research purpose and how you plan to use the dataset
Include your affiliation and contact information

All requests will be reviewed, and access will be granted for academic and research purposes in accordance with our data usage policy.

Dataset page: https://sites.google.com/view/multi-emp?usp=sharing
