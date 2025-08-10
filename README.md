Project Summary

In line with natIgnite’s 2025 theme of AgeTech, our team developed a teleoperation solution aimed at assisting elderly individuals living with rheumatoid arthritis. The project centers on a 3D-printed robotic hand designed to perform common gestures and grips with minimal physical strain on the user.

Our primary control method uses electromyography (EMG) signals to detect muscle contractions and relaxations. These signals are captured using the BioAmp EXG Pill connected to an Arduino, which processes and classifies them into pre-programmed gestures. For instance, a strong muscle clench can trigger a firm grip, while a quick clench-release sequence can toggle between grasp and release. This setup allows users with limited mobility to perform actions without complex mechanical controls.

In parallel, we implemented a voice-command interface. Spoken instructions are interpreted by a large language model (LLM), which can match the request to an existing gesture or generate entirely new servo configurations. These new configurations can then be saved and reused, enabling adaptive personalization over time.

By combining EMG and voice control, our system offers flexibility for users with different abilities and preferences. The teleoperation capability means the robotic hand can be operated remotely, opening the door for future applications in home assistance, rehabilitation, and remote caregiving.
