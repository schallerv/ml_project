import streamlit as st
from setup import BACKGROUND_COLOR, PRIMARY_COLOR, SECONDARY_COLOR, TERNARY_COLOR, BLACK_COLOR

# st.set_page_config(page_title="Project Explanation")

def explanation_app():
    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background-color: {BACKGROUND_COLOR};
            color: {BLACK_COLOR};
        }}
        .sidebar .sidebar-content {{
            background-color: {BACKGROUND_COLOR};
        }}
        .stTextInput>div>input {{
            background-color: {SECONDARY_COLOR};
            color: {BLACK_COLOR};
        }}
        div.stButton > button {{
            background-color: {PRIMARY_COLOR};
            color: {BLACK_COLOR};
            border: none;
            padding: 0.5em 1em;
            border-radius: 5px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Project Explanation and Analysis")
    st.header("High-Level Project Description")
    st.title("Project Explanation and Analysis")

    # High-Level Project Description
    st.header("High-Level Project Description")
    st.markdown("""
            This project leverages deep learning and content-based filtering techniques to recommend board games tailored to user preferences.
            We work with a very high-dimensional feature set—over 4000 game features that include more than 3000 binary flags along with 
            continuous attributes. In addition, we incorporate semantic information through sBERT embeddings computed from game descriptions.
            These diverse data are processed through a noising autoencoder which extracts a dense, robust latent representation.  
            The latent features then power our content-based filtering system, which compares games based on cosine similarity.
            """)

    # Data Source and Feature Construction
    st.header("Data Source and Feature Construction")
    st.markdown("""
            **Data Source:**  
            We use the [Board Games Database from BoardGameGeek on Kaggle](https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek),  
            which includes a wealth of data on board games—ranging from ratings and play times to qualitative metadata.

            **Types of Game Features:**  
            - **Core Game Data:** Includes numerical features such as year published, average rating, play time, etc.  
              These features provide direct quantitative insight into the game’s performance and complexity.  
            - **Binary Flags for Categories and Themes:** From the BGG documentation, you will find features such as game themes, mechanics, subcategories,  
              and even designer, artist, and publisher indicators. These binary flags help capture the stylistic and categorical aspects of games.  
              For example, knowing that two games share similar themes (e.g., strategy or war) can be highly useful when computing similarity.  
            - **sBERT Embeddings:** To enrich our feature set, we compute sBERT embeddings from game descriptions.  
              These semantic embeddings are concatenated with the numerical and binary features, enabling the model to capture unstructured textual nuances  
              alongside structured data.  

            Together, these varied feature types allow for a more nuanced and comprehensive representation of each game,
            which is critical for effective content-based filtering.
            """)

    # Noising Autoencoder Section
    st.header("Noising Autoencoder")
    st.markdown("""
            **Overview:**  
            The noising autoencoder is designed to learn a robust latent representation of our high-dimensional board game features.
            Given that we have over 4000 input features (including more than 3000 binary flags), a dense representation is necessary to 
            capture the essential aspects of the data. During training, Gaussian noise is added to the inputs so the autoencoder learns 
            to recover the noise-free version, thereby reinforcing the extraction of noise-invariant and important features.
            """)

    st.markdown("### Loss Curve Analysis")
    st.markdown("""
            The loss curve below illustrates the progression of training and test loss over the epochs.  
            Notice that the autoencoder’s loss converged and shows no signs of overfitting, indicating that the model
            successfully learned to reconstruct the input data despite its high dimensionality.
            """)
    st.image("loss_curve.png", caption="Loss Curve: Training and Test Loss over Epochs")

    st.markdown("### Custom Loss Function")
    st.markdown("""
            **Motivation:**  
            Our input feature vector is heterogeneous, composed of:
            - Continuous numerical features (columns 0 to 513)
            - Binary flags (the remaining columns)

            **Implementation:**  
            We created a custom loss function (`MSE_BCE_Loss`) that computes:
            - **Mean Squared Error (MSE):** for the continuous features.
            - **Binary Cross-Entropy (BCE) Loss:** for the binary features.

            **Weighting Strategy:**  
            The losses are weighted differently (for example, using a weight of 1 for MSE and 0.5 for BCE) to ensure that errors in the
            larger-scale continuous data do not dominate the training process. This differential weighting is crucial to balance the reconstruction of both data types.

            **Why It Was Necessary:**  
            A single loss metric (e.g., pure MSE) would treat binary flags as continuous data and potentially yield poor predictions for
            classification-like tasks. The weighted hybrid loss ensures that the autoencoder learns accurate and robust representations
            for both kinds of features.
            """)

    # Neighbor Preservation Analysis (Embedding Comparison removed)
    st.header("Neighbor Preservation Analysis")
    st.markdown("""
            **Overview:**  
            We evaluate the quality of the latent space by checking if the local relationships among games are maintained.
            The analysis shows that many of the original neighbors in the high-dimensional space are preserved in the latent space,
            confirming that the autoencoder effectively captured the essential structure of the data.
            """)
    st.image("neighborhood_preservation.png",
             caption="Neighbor Preservation: A majority of game neighbors are maintained.")

    # Content-Based Filtering Section
    st.header("Content-Based Filtering")
    st.markdown("""
            **Overview:**  
            The content-based filtering mechanism leverages the latent representations from the autoencoder to recommend games with similar content.

            **Algorithm Details:**  
            - **Latent Embeddings:** Each game is mapped to a low-dimensional latent space.
            - **Similarity Measurement:** Cosine similarity (with scores normalized to range from 0 to 1) is computed between game embeddings.
            - **Rating-Based Weighting:** Historical user ratings are used to weight the similarity scores when predicting preferences.
            - **Top‑*k* Aggregation:** The recommendation for a user is based on the top‑*k* similar games.

            **Observations:**  
            Though the system tends to overestimate very lowly rated games and underestimate very highly rated ones, the overall ranking
            of games is preserved. The true versus predicted plot further demonstrates that games with higher ratings are generally predicted 
            as such, and vice versa.

            **Cold Start Considerations:**  
            Content-based filtering addresses the cold start problem differently than collaborative filtering. Unlike user–user or item–item methods,
            which rely on historical interactions, content-based filtering uses game attributes to recommend new or unrated games.  
            This means even if a game lacks extensive rating history, its content features can still be used to assess similarity and generate recommendations.
            """)

    st.markdown("### True vs. Predicted Ratings Analysis")
    st.markdown("""
            The true versus predicted plot compares actual user ratings with those predicted by our recommendation model.
            While extreme ratings are somewhat compressed (low ratings are overestimated and high ratings underestimated),
            the overall ordering of games is maintained—demonstrating that higher rated games generally receive higher predicted ratings.
            """)
    st.image("true_vs_predicted.png", caption="True vs Predicted Ratings: Overall ordering is preserved.")

    # Conclusion and Future Work
    st.header("Conclusion and Future Work")
    st.markdown("""
            In summary, our project successfully combines a noising autoencoder with content-based filtering to generate personalized board game recommendations.
            The autoencoder compresses a very high-dimensional, heterogeneous feature space into a robust latent representation, as evidenced by
            the converged loss curve and strong neighbor preservation. The custom loss function—with its balanced weighting between MSE and BCE—
            enables the model to handle both continuous and binary features effectively.

            Our content-based filtering method, despite some bias in predicting extreme ratings, preserves the general ranking of games and overcomes the cold start problem by
            focusing on content features rather than historical interaction data.

            **Future Work:**  
            Future enhancements could explore advanced models such as set transformers or graph transformers, which may better capture complex relationships 
            in the data and further improve recommendation accuracy.
            """)

if __name__ == "__main__":
    explanation_app()
