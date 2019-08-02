## Udacity AWS DeepRacer Scholarship Challenge

*AWS and Udacity are teaming up to teach machine learning and prepare students to test their skills by participating in the world’s first autonomous racing league—the AWS DeepRacer League. Students with the top lap times will earn full scholarships to the Machine Learning Engineer Nanodegree program.*

* https://www.udacity.com/aws-deepracer-scholarship

## AWS Deep Racer Locally:

1. https://github.com/alexschultz/deepracer-for-dummies
2. Run `./init.sh` to initialize
3. Important files:
  * Training hyperparameters:  `rl_deepracer_coach_robomaker.py`
  * Action Space: `docker/volumes/minio/bucket/custom_files/model_metadata.json`
  * Reward Function: `docker/volumes/minio/bucket/custom_files/reward.py`
  * Track selection: `docker/.env`  
