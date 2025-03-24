# pipeline.py

def run_pipeline():
    print("\033[1;34m[Step 1] Data Analysis\033[0m")
    import data_analysis

    print("\033[1;34m[Step 2] Data Preprocessing\033[0m")
    import data_modification

    print("\033[1;34m[Step 3] Model Training and Evaluation\033[0m")
    import training

    print("\033[1;34m[Step 4] Optimization, Test Prediction & Submission\033[0m")
    import optimization

    print("\033[1;32mPipeline complete.\033[0m")


if __name__ == "__main__":
    run_pipeline()
