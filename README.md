# RECOMMENDATION-SYSTEM-APPLICATION

COMPANY : CODTECH IT SOLUTIONS

NAME : R GIRIDHAR

INTERN ID : CTIS2163

DOMAIN : MACHINE LEARNING 

DURATION : 4 WEEKS

MENTOR : NEELA SANTOSH

Movie Recommendation System Web Application
 
Project Overview

The Movie Recommendation System Web Application is a machine learning–powered system designed to deliver personalized movie recommendations using collaborative filtering techniques. Recommendation systems play a crucial role in modern digital platforms by helping users discover relevant content efficiently. This project demonstrates how user preferences can be learned from historical interaction data and used to predict future interests through matrix factorization.
The application is built as an interactive web interface using Streamlit, making it easy for users to explore recommendations without requiring technical expertise. The backend leverages Singular Value Decomposition (SVD), a widely used matrix factorization technique, to identify hidden patterns in user–movie interactions. The system is trained and evaluated using the MovieLens 100K dataset, a benchmark dataset commonly used in academic research and industry experiments.

System Architecture and Methodology

The core of the system is based on collaborative filtering, which assumes that users who have shown similar behavior in the past will have similar preferences in the future. Instead of relying on movie metadata such as genre or cast, the model uses only historical ratings to learn user behavior patterns.
Matrix factorization through SVD decomposes the user–item interaction matrix into lower-dimensional latent feature vectors representing users and movies. These latent features capture underlying characteristics such as user taste and movie popularity. By reconstructing the matrix, the system predicts ratings for movies that a user has not yet rated, enabling personalized recommendations.
The model is trained on 80% of the dataset and evaluated on the remaining 20%. This train–test split ensures reliable performance assessment while preventing overfitting. Once trained, the system generates top-N movie recommendations by ranking predicted ratings in descending order.

Technologies Used

The application is implemented using Python and incorporates several popular data science and machine learning libraries. Scikit-Surprise is used to implement the recommendation algorithms, while Pandas and NumPy handle data processing. Streamlit provides a lightweight and interactive framework for building the web interface, enabling real-time interaction with the recommendation system.

Evaluation and Performance Metrics

To measure the accuracy of the recommendation model, two standard evaluation metrics are used: Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE). RMSE penalizes larger prediction errors and provides insight into overall model accuracy, while MAE measures the average absolute difference between predicted and actual ratings. Lower values for both metrics indicate better performance. These metrics are displayed directly in the application to provide transparency and performance insights.

Application Features

The web application allows users to input a valid user ID and select the number of recommendations they wish to receive. The system then generates a list of recommended movies along with their predicted ratings. The interface also displays model evaluation metrics, offering users an understanding of the system’s reliability. The design is simple, intuitive, and suitable for both technical and non-technical users.

Conclusion

This Movie Recommendation System demonstrates the practical application of collaborative filtering and matrix factorization in building intelligent, user-centric systems. By combining machine learning techniques with an interactive web interface, the project highlights how recommendation systems are implemented in real-world platforms such as Netflix and Amazon. The system provides a strong foundation for further enhancements, including hybrid recommendation approaches, deployment at scale, and integration with real-time user data.

<img width="1920" height="1080" alt="1" src="https://github.com/user-attachments/assets/81d51c1a-f0ec-4d9a-b303-b86305648e42" />


<img width="1920" height="1080" alt="2" src="https://github.com/user-attachments/assets/d91fa43a-89dd-4fbe-a925-887c0740acc4" />


