## Udacity AWS DeepRacer Scholarship Challenge

*AWS and Udacity are teaming up to teach machine learning and prepare students to test their skills by participating in the world’s first autonomous racing league—the AWS DeepRacer League. Students with the top lap times will earn full scholarships to the Machine Learning Engineer Nanodegree program.*

* https://www.udacity.com/aws-deepracer-scholarship

## Notes on training new models for the `Shanghai Sudu` track (August 2019)

* All submissions should use PPO algorithm to train for the track.
* The implementation is with the Tensorflow Library
* Penalizing for slow speed seems to work.
* I tried a simple reward function (You can find it in the current folder).
* Increasing the granularity seems to help
* Initial speed of 2 m/s
* 1 hour of training wasn't enough with my current reward function.
* With 2 hours of training the track was completed and did a succesful submission.
* With 3 hours I Noticed further improvement.

Model details:
* Speed `2 m/s` and `3 m/s`
* Max granularity (Both actions and speed)
* Batch size `64`
* Entropy `0.01`
* Discount factor `0.8`
* Loss type `Huber`
* Learning rate: `0.0006`
* Epochs `10`
* Number of experience episodes before updating policy `20`.

## AWS Deep Racer Locally:

1. https://github.com/alexschultz/deepracer-for-dummies
2. Run `./init.sh` to initialize
3. Important files:
  * Training hyperparameters:  `rl_deepracer_coach_robomaker.py`
  * Action Space: `docker/volumes/minio/bucket/custom_files/model_metadata.json`
  * Reward Function: `docker/volumes/minio/bucket/custom_files/reward.py`
  * Track selection: `docker/.env`  
