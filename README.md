# ML for Robotic Fabrication - Pythagoras Project

## Dataset Description

This project generates a synthetic dataset of right-angle triangles using random leg lengths (`leg_a`, `leg_b`) and calculates the hypotenuse using the Pythagorean theorem. The dataset is saved in CSV format in the `data/` folder.

- File: `data/triangles.csv`
- Columns: `leg_a`, `leg_b`, `hypotenuse`

## Model

A simple linear regression model is trained using `scikit-learn` to predict the hypotenuse from the two legs of the triangle.

### Libraries Used
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

##  Evaluation

- **Metric 1:** Mean Squared Error (MSE)
- **Metric 2:** R² Score
- **Visualization:** A scatter plot comparing true vs predicted hypotenuse values is saved in `src/hypotenuse_plot.png`.

##  How to Run

1. **Generate the dataset:**

```bash
python3 src/generate_data.py
python3 src/train_model.py
