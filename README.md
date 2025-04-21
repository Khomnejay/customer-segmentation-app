*Customer Segmentation App

      This is an interactive web application built using Streamlit that performs customer segmentation using RFM analysis (Recency, Frequency, Monetary) and             KMeans clustering. It is designed to help businesses understand customer behavior from retail transaction data.

*Features

      Upload and preprocess online retail data.
      
      Perform RFM analysis to evaluate customer value.
      
      Apply KMeans clustering to segment customers into different groups.
      
      Visualize clusters using PCA and interactive Seaborn/Matplotlib plots.
      
      Real-time exploration and customization through a user-friendly Streamlit interface.


Tech Stack

      Python
      
      Pandas, NumPy
      
      Scikit-learn
      
      Streamlit
      
      Matplotlib, Seaborn


Getting Started

      1. Clone the repository:
      
      git clone https://github.com/your-username/customer-segmentation-app.git
      cd customer-segmentation-app
      
      
      2. Install dependencies:
      
      pip install -r requirements.txt
      
      
      3. Run the app:
      
      streamlit run app.py
      
      
      4. Upload your online_retail.csv file when prompted or place it in the root directory.



Dataset

      The app expects a dataset similar to the UCI Online Retail dataset, containing fields such as:
      
      InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country


Screenshots
         ![image](https://github.com/user-attachments/assets/678fe90a-1040-42c3-919f-786994b83f59)
         ![image](https://github.com/user-attachments/assets/d57829b1-2bdb-4881-9c06-04c88b7e26f0)
         ![image](https://github.com/user-attachments/assets/bba88a15-1170-4dfe-b47d-2b69d8bc03b8)
         ![image](https://github.com/user-attachments/assets/c49ed74d-9d35-40ea-83de-611f70de5c65)





License

        This project is open-source and available under the MIT License.
