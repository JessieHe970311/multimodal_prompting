### How to run the system
1. Download image data (extracted_frames, images, segmented_videos_refine) [here]() to ``./frontend/src/assets/``.
2. Install Python packages (suggest using conda for package management, python3.9 is suggested):
   ```
   cd backend
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API-KEY through the terminal.
   ```
   export OPENAI_API_KEY="$your_key"
   ```
4. Set up backend.
   ```
   python run-data-backend.py
   ```
6. Set up frontend. 
   ```
   cd frontend
   npm install
   npm run serve 
   ```