import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict
import matplotlib.pyplot as plt
import io
from scipy.stats import f
from statsmodels.stats.stattools import durbin_watson
from fpdf import FPDF
import math
import itertools

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Test de Permutation pour Modèles QSAR",
    layout="centered"
)

st.title("Test de Permutation (Y-Scrambling) pour Modèles QSAR")
st.markdown("""
Cette application évalue la **significativité** d’un modèle de régression linéaire multiple (QSAR) 
à l’aide d’un **test de permutation**. Elle permet de déterminer si la relation observée entre les 
descripteurs (X) et l’activité (y) est réelle ou simplement due au hasard, en comparant le $R^2$ 
**original** à une distribution de $R^2$ **obtenue** après permutations aléatoires de la variable réponse 'y'.

Cet outil réalise un **test de permutation (Y-scrambling)** associé à une **validation Leave-One-Out (LOO)** 
afin de vérifier la robustesse et la significativité statistique du modèle QSAR.
""")

# --- Fonction de Parsing des Données ---
@st.cache_data
def load_data(data_string):
    """Charge et nettoie les données à partir d'une chaîne de caractères."""
    # Utilise io.StringIO pour lire la chaîne comme un fichier
    try:
        # Lecture du CSV
        df = pd.read_csv(io.StringIO(data_string.strip()), sep=',', skipinitialspace=True)
        
        # Vérification des colonnes
        if 'y' not in df.columns:
            st.error("Erreur: La colonne de la variable réponse 'y' est manquante.")
            return None, None
            
        X_cols = [col for col in df.columns if col.startswith('X')]
        if not X_cols:
            st.error("Erreur: Aucune colonne de descripteurs (nommée 'X1', 'X2', etc.) n'a été trouvée.")
            return None, None

        # Conversion et nettoyage
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        
        if df.empty:
            st.error("Erreur: Le DataFrame est vide après le nettoyage (vérifiez les en-têtes et le format des nombres).")
            return None, None
            
        return df, X_cols
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des données. Détail : {e}")
        return None, None

# --- Saisie des données par l'utilisateur ---
st.header("1. Entrée des Données Moléculaires")

input_method = st.radio(
    "Choisissez votre méthode d'entrée des données :",
    ("Copier-coller le texte", "Uploader un fichier CSV"),
    horizontal=True
)

df = None
X_cols = None

if input_method == "Copier-coller le texte":
    # Données par défaut basées sur votre fichier
    default_data = """
    X1,X2,y
    1.48,1.340,0.34
    1.40,1.339,0.29
    2.08,1.339,0.41
    2.26,1.339,0.50
    1.26,1.340,0.15
    2.29,1.338,0.59
    3.62,1.339,0.70
    2.65,1.340,0.97
    """

    st.info("Veuillez coller vos données au format CSV, avec une ligne d'en-tête. Les colonnes doivent être nommées 'X1', 'X2', ... pour les descripteurs et **'y'** pour la variable réponse.")
    data_input = st.text_area(
        "Collez vos données ici :",
        default_data,
        height=200
    )
    df, X_cols = load_data(data_input)

elif input_method == "Uploader un fichier CSV":
    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
    if uploaded_file is not None:
        # Pour lire le fichier uploadé, on le décode d'abord
        string_data = uploaded_file.getvalue().decode("utf-8")
        df, X_cols = load_data(string_data)

if df is not None:
    st.subheader("Aperçu des Données Utilisées")
    st.dataframe(df)

    # Initialiser l'état de session pour conserver les résultats
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False

    # --- Paramètres de l'Analyse ---
    st.header("2. Paramètres du Test")
    col1, col2 = st.columns(2)
    
    n_permutations = col1.number_input(
        "Nombre de Permutations (N)",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="Plus ce nombre est grand, plus la p-value sera précise."
    )
    
    random_seed = col2.number_input(
        "Générateur Aléatoire (Seed)",
        min_value=0,
        value=42,
        step=1,
        help="Permet de garantir la reproductibilité des résultats."
    )

    # Seuil pour le test exact
    n = len(df)
    try:
        total_perms = math.factorial(n)
    except ValueError:
        total_perms = float('inf')
    

    # --- Fonctions de calcul ---
    def q2_loo(modele, X, y):
        """
        Calcule le Q² (LOO) de manière optimisée pour la régression linéaire.
        Cette méthode évite de ré-entraîner le modèle n fois.
        """
        # S'assurer que X a une colonne de 1s pour l'intercept
        X_with_intercept = np.c_[np.ones(X.shape[0]), X]
        
        # Entraîner le modèle une seule fois
        modele.fit(X, y)
        y_pred_full = modele.predict(X)
        residuals = y - y_pred_full
        
        # Calculer la diagonale de la matrice chapeau (hat matrix)
        try:
            hat_diag = np.sum(X_with_intercept * (np.linalg.pinv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T).T, axis=1)
        except np.linalg.LinAlgError:
            # Fallback si le calcul est instable
            return q2_loo_fallback(modele, X, y)

        # Calculer les résidus de validation croisée (PRESS)
        press_residuals = residuals / (1 - hat_diag)
        ss_press = np.sum(press_residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - ss_press / ss_tot

    # --- Fonction du Test de Permutation ---
    def permutation_test(X, y, n_permutations, random_seed):
        """Effectue le test de permutation sur le Q² et retourne le Q² original et les Q² permutés."""
        model = LinearRegression()
        q2_original = q2_loo(model, X, y)

        # 2. Initialisation
        q2_permuted = []
        progress_bar = st.progress(0)

        # 3. Choix entre test exact et aléatoire
        if total_perms <= 50000: # Seuil pour le test exact
            st.info(f"Test exact en cours : évaluation des {total_perms} permutations possibles.")
            perms = list(itertools.permutations(y))
            for i, perm_indices in enumerate(itertools.permutations(range(len(y)))):
                y_perm = y.iloc[list(perm_indices)].values
                q2_permuted.append(q2_loo(model, X, y_perm))
                progress_bar.progress((i + 1) / total_perms)
        else:
            st.info(f"Test par échantillonnage aléatoire ({n_permutations} permutations).")
            np.random.seed(random_seed)
            for i in range(n_permutations):
                y_shuffled = np.random.permutation(y)
                q2_permuted.append(q2_loo(model, X, y_shuffled))
                
                # Mise à jour de la barre de progression
                progress_bar.progress((i + 1) / n_permutations)

        progress_bar.empty()

        return q2_original, np.array(q2_permuted)

    # --- Fonction pour trouver le R² maximal par permutation ---
    def find_max_r2_permutation(X, y, n_random_perms=5000):
        """
        Trouve le R² maximal possible en permutant y.
        - Test exact si n <= 9.
        - Test par échantillonnage aléatoire sinon.
        """
        model = LinearRegression()
        n = len(y)
        max_r2 = -1
        best_permutation = None

        try:
            total_perms = math.factorial(n)
        except ValueError:
            total_perms = float('inf')

        if n <= 9: # Seuil pour le test exact
            st.info(f"Recherche du R² maximal (test exact sur {total_perms} permutations)...")
            for y_perm_tuple in itertools.permutations(y):
                y_perm = np.array(y_perm_tuple)
                model.fit(X, y_perm)
                current_r2 = r2_score(y_perm, model.predict(X))
                if current_r2 > max_r2:
                    max_r2 = current_r2
                    best_permutation = y_perm
        else: # Test par échantillonnage pour les n plus grands
            st.info(f"Recherche du R² maximal (estimation sur {n_random_perms} permutations aléatoires)...")
            for _ in range(n_random_perms):
                y_shuffled = np.random.permutation(y)
                model.fit(X, y_shuffled)
                current_r2 = r2_score(y_shuffled, model.predict(X))
                if current_r2 > max_r2:
                    max_r2 = current_r2
                    best_permutation = y_shuffled
        
        return max_r2, best_permutation

    # --- Fonction de génération de rapport PDF ---
    def create_pdf_report(df, perm_results_df, anova_df, error_metrics_df, residuals_df, summary_df, fig):
        """Génère un rapport PDF complet de l'analyse."""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        
        # Titre
        pdf.cell(0, 10, "Rapport du Test de Permutation QSAR", 0, 1, "C")
        pdf.ln(10)

        def write_df_to_pdf(pdf, title, dataframe):
            """Écrit un DataFrame pandas dans le PDF avec gestion du retour à la ligne."""
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, title, 0, 1, "L")
            pdf.set_font("Arial", "B", 9)
            
            # Calcul des largeurs de colonnes
            available_width = pdf.w - pdf.l_margin - pdf.r_margin
            num_cols = len(dataframe.columns)
            # 40% pour l'index, 60% pour le reste
            index_col_width = available_width * 0.4
            data_col_width = (available_width * 0.6) / num_cols if num_cols > 0 else 0
            
            col_widths = [index_col_width] + [data_col_width] * num_cols
            header = [dataframe.index.name or 'Index'] + dataframe.columns.tolist()

            # Header
            for i, col_name in enumerate(header):
                pdf.cell(col_widths[i], 7, str(col_name), 1, 0, 'C')
            pdf.ln()

            # Body
            pdf.set_font("Arial", "", 9)
            for index, row in dataframe.iterrows():
                row_data = [str(index)] + [str(item) for item in row]
                
                # Calculer la hauteur de ligne max nécessaire pour cette ligne
                line_height = pdf.font_size * 1.5
                max_lines = 1
                for i, datum in enumerate(row_data):
                    lines = pdf.multi_cell(col_widths[i], line_height, datum, border=0, dry_run=True, output='lines')
                    max_lines = max(max_lines, len(lines))
                row_height = line_height * max_lines

                # Dessiner les cellules avec la hauteur calculée
                for i, datum in enumerate(row_data):
                    pdf.multi_cell(col_widths[i], line_height, datum, border=1, new_x="RIGHT", new_y="TOP")
                
                pdf.ln(row_height)

            pdf.ln(5)

        # Écrire les sections
        write_df_to_pdf(pdf, "Resultats du Test de Permutation", perm_results_df)
        write_df_to_pdf(pdf, "Analyse de la Variance (ANOVA)", anova_df.applymap(lambda x: f'{x:.4f}' if isinstance(x, (int, float)) else x))
        write_df_to_pdf(pdf, "Statistiques des Erreurs et de Regression", error_metrics_df)

        # Image
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Distribution du Q2 des Modeles Permutes", 0, 1, "L")
        with io.BytesIO() as buffer:
            fig.savefig(buffer, format="png", dpi=300)
            pdf.image(buffer, x=pdf.get_x() + 30, w=pdf.w - 100)
        pdf.ln(5)

        # Données d'entrée
        pdf.add_page()
        write_df_to_pdf(pdf, "Donnees d'entree", df.head(20)) # Limite aux 20 premières lignes

        # Analyse des résidus
        write_df_to_pdf(pdf, "Analyse des Residus", residuals_df.head(20).round(4))

        # Guide d'interprétation
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Guide d'Interpretation des Metriques", 0, 1, "L")
        
        available_width = pdf.w - pdf.l_margin - pdf.r_margin
        for i, row in summary_df.iterrows():
            pdf.set_font("Arial", "B", 9)
            pdf.multi_cell(available_width, 5, f"{i}: {row['Utilite']}")
            pdf.set_font("Arial", "", 9)
            pdf.multi_cell(available_width, 5, f"    Interpretation ideale: {row['Interpretation Ideale']}")
            pdf.ln(2)

        return bytes(pdf.output(dest='S'))

    # --- Exécution et Affichage des Résultats ---
    if st.button("Lancer le Test de Permutation 🚀", key="run_test", use_container_width=True):
        X = df[X_cols]
        y = df['y']
        
        spinner_message = f"Exécution des {n_permutations} permutations..."
        if total_perms <= 50000:
            spinner_message = f"Exécution du test exact ({total_perms} permutations)..."

        with st.spinner(spinner_message):
            q2_original, q2_permuted = permutation_test(X, y, n_permutations, random_seed)
            max_r2_perm, best_perm_y = find_max_r2_permutation(X, y)
        
        # Calcul du p-value: proportion des Q² permutés >= Q² original
        q2_count_higher = np.sum(q2_permuted >= q2_original)
        
        # Le dénominateur dépend du type de test
        num_total_tests = len(q2_permuted)
        p_value = q2_count_higher / num_total_tests

        # --- Calcul des métriques additionnelles sur le modèle original ---
        model_orig = LinearRegression()
        model_orig.fit(X, y)
        y_pred_orig = model_orig.predict(X)
        
        r2_original = r2_score(y, y_pred_orig)

        n_obs = len(y)
        p_vars = X.shape[1]
        
        # --- Calculs pour ANOVA et autres métriques ---
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean)**2)
        ss_res = np.sum((y - y_pred_orig)**2)
        ss_reg = ss_tot - ss_res

        df_reg = p_vars
        df_res = n_obs - p_vars - 1
        
        ms_reg = ss_reg / df_reg if df_reg > 0 else 0
        ms_res = ss_res / df_res if df_res > 0 else 0
        
        f_stat = ms_reg / ms_res if ms_res > 0 else 0
        f_p_value = f.sf(f_stat, df_reg, df_res) if df_res > 0 else np.nan

        # R² ajusté
        if df_res > 0:
            r2_adj = 1 - (1 - r2_original) * (n_obs - 1) / (n_obs - p_vars - 1)
        else:
            r2_adj = np.nan # Non défini

        # Erreurs
        mse = mean_squared_error(y, y_pred_orig)
        rmse = np.sqrt(mse) # Aussi appelé "Standard Error of the Estimate"

        # Durbin-Watson
        residuals = y - y_pred_orig
        dw_stat = durbin_watson(residuals)
        intercept = model_orig.intercept_

        # --- Création des DataFrames pour les tableaux ---

        # 1. Tableau ANOVA
        anova_data = {
            'Source': ['Régression', 'Résidus', 'Total'],
            'DDL': [df_reg, df_res, n_obs - 1],
            'Somme des Carrés (SS)': [ss_reg, ss_res, ss_tot],
            'Moyenne des Carrés (MS)': [ms_reg, ms_res, None],
            'F-Statistique': [f_stat, None, None],
            'Signif. F': [f_p_value, None, None]
        }
        anova_df = pd.DataFrame(anova_data).set_index('Source')

        # 2. Tableau des erreurs
        error_metrics_df = pd.DataFrame({
            "Metrique": ["R²", "R² Ajusté", "Erreur Std. de l'Estimation (RMCE)", "Statistique de Durbin-Watson", "Observations"],
            "Valeur": [f"{r2_original:.4f}", f"{r2_adj:.4f}" if not np.isnan(r2_adj) else "N/A", f"{rmse:.4f}", f"{dw_stat:.4f}", str(n_obs)]
        }).set_index("Metrique")

        # 3. Tableau des résidus
        residuals_df = pd.DataFrame({
            'Valeur Observee (y)': y.values,
            'Valeur Predite (y_pred)': y_pred_orig,
            'Residu (y - y_pred)': residuals.values
        })
        
        # 4. Tableau des résultats de permutation
        perm_results_df = pd.DataFrame({
            "Métrique de Permutation": ["Q² observé (LOO)", "Moyenne Q² permutés", "Max Q² permutés", "p-value (permutation)", "R² max. par permutation"],
            "Valeur": [f"{q2_original:.4f}", f"{np.mean(q2_permuted):.4f}", f"{np.max(q2_permuted):.4f}", f"{p_value:.4f}", f"{max_r2_perm:.4f}"]
        }).set_index("Métrique de Permutation")

        # --- Création du Graphique ---
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogramme
        ax.hist(q2_permuted, bins=30, color='skyblue', edgecolor='black', alpha=0.7, label='Q² Modèles Permutés')
        
        # Q2 original
        ax.axvline(q2_original, color='red', linestyle='--', linewidth=2, label=f'Q² Original: {q2_original:.4f}')
        
        # Ombrage pour la p-value
        min_shade = max(q2_original, q2_permuted.min() - 0.01) 
        ax.axvspan(min_shade, q2_permuted.max(), color='red', alpha=0.1, label='Zone de la p-value (Q² >= Q² Original)')

        ax.set_title(f'Distribution des $Q^2$ des Modèles Permutés (N={num_total_tests})')
        ax.set_xlabel('$Q^2$ (Leave-One-Out)')
        ax.set_ylabel('Fréquence')
        ax.legend()
        ax.grid(axis='y', alpha=0.5)

        # --- Tableau récapitulatif final ---
        summary_data = {
            "Metrique": ["Q2 observe (LOO)", "p-value (permutation)", "R2", "R2 Ajusté", "F-Statistique (ANOVA)", "Signif. F (ANOVA)", "Erreur Std. de l'Estimation (RMCE)", "Statistique de Durbin-Watson"],
            "Utilite": ["Mesure la capacite predictive du modele sur de nouvelles donnees (robustesse).", "Probabilite que la performance observee (Q2) soit due au hasard.", "Mesure la proportion de la variance de 'y' expliquee par le modele (qualite de l'ajustement).", "Similaire au R2, mais penalise l'ajout de variables inutiles.", "Teste si au moins un des predicteurs est significativement lie a la variable reponse.", "La p-value associee au test F. Indique la significativite globale du modele.", "L'ecart-type des residus. Indique la magnitude typique de l'erreur de prediction.", "Detecte l'autocorrelation des residus. Une hypothese cle de la regression est leur independance."],
            "Interpretation Ideale": ["Le plus eleve possible (proche de 1). > 0.5 est souvent considere comme bon.", "Le plus bas possible (< 0.05).", "Le plus eleve possible (proche de 1).", "Proche du R2, indiquant que les variables sont utiles.", "Le plus eleve possible.", "Le plus bas possible (< 0.05).", "Le plus bas possible.", "Proche de 2. Des valeurs << 2 ou >> 2 indiquent un probleme."]
        }
        summary_df = pd.DataFrame(summary_data).set_index("Metrique")

        # --- Stockage des résultats dans l'état de session ---
        st.session_state.analysis_done = True
        st.session_state.results = {
            "r2_original": r2_original,
            "p_value": p_value,
            "q2_count_higher": q2_count_higher,
            "num_total_tests": num_total_tests,
            "perm_results_df": perm_results_df,
            "anova_df": anova_df,
            "error_metrics_df": error_metrics_df,
            "residuals_df": residuals_df,
            "summary_df": summary_df,
            "fig": fig,
            "best_perm_y": best_perm_y,
            "df_input": df # Sauvegarde des données d'entrée pour le rapport
        }

    # --- Affichage des résultats (si l'analyse a été faite) ---
    if st.session_state.analysis_done:
        # Récupération des résultats depuis l'état de session.
        # C'est la ligne qui manquait.
        results = st.session_state.results 
        
        st.header("3. Résultats de l'Analyse")
        # --- Affichage des Métriques Clés ---
        st.subheader("Synthèse des Résultats")
        
        col_res1, col_res2 = st.columns(2)
        
        col_res1.metric(
            label="R² du Modèle Original",
            value=f"{results['r2_original']:.4f}"
        )

        col_res2.metric(
            label="P-Value Calculée",
            value=f"{results['p_value']:.4f}",
            delta="SIGNIFICATIF" if results['p_value'] < 0.05 else "NON SIGNIFICATIF",
            delta_color="inverse"
        )
        
        st.markdown(f"""
        * **P-Value**: `{results['p_value']:.4f}`.
        * **Interprétation**: {results['q2_count_higher']} des {results['num_total_tests']} modèles permutés ont généré un $Q^2$ supérieur ou égal au $Q^2$ original.
        * Si **P-Value < 0.05**, le modèle est considéré comme **significatif** ; le $R^2$ n'est pas dû au hasard.
        """)

        # --- Affichage des tableaux de statistiques détaillées ---
        st.subheader("Analyse Détaillée du Modèle Original")

        st.markdown("##### Résultats du Test de Permutation")
        st.table(results['perm_results_df'])
        st.markdown("Ce tableau évalue la robustesse du modèle. Un bon modèle a un `Q² observé` bien supérieur à la `Moyenne Q² permutés`. Le `R² max. par permutation` montre le meilleur score atteignable par pur hasard.")

        st.markdown("##### Analyse de la Variance (ANOVA)")
        st.table(results['anova_df'].style.format({
            'Somme des Carrés (SS)': '{:.4f}', 
            'Moyenne des Carrés (MS)': '{:.4f}',
            'F-Statistique': '{:.4f}',
            'Signif. F': '{:.4f}'
        }, na_rep=""))
        st.markdown("Le test F évalue la significativité globale du modèle. Une valeur `Signif. F` < 0.05 indique que le modèle est statistiquement significatif.")

        st.markdown("##### Statistiques des Erreurs et de Régression")
        st.table(results['error_metrics_df'])
        st.markdown("La statistique de Durbin-Watson teste l'autocorrélation des résidus. Une valeur proche de 2 indique une absence d'autocorrélation.")

        st.markdown("##### Analyse des Résidus")
        st.dataframe(results['residuals_df'].style.format('{:.4f}'))
        st.markdown("L'analyse des résidus permet d'identifier les observations pour lesquelles le modèle est moins performant.")


        # --- Affichage du Graphique ---
        st.subheader("Distribution du Q² des Modèles Permutés")
        st.pyplot(results['fig'])
        
        # --- Tableau récapitulatif final ---
        st.subheader("Guide d'Interprétation des Métriques")
        st.table(results['summary_df'])

        # --- Section explicative sur le R² max par permutation ---
        with st.expander("🔍 À propos du 'R² maximal par permutation'"):
            
            # Création d'un DataFrame pour comparer y original et y permuté
            perm_comparison_df = pd.DataFrame({
                'y Original': results['df_input']['y'],
                'Meilleure Permutation (y permuté)': results['best_perm_y']
            })
            perm_comparison_df['y Original'] = perm_comparison_df['y Original'].round(4)
            perm_comparison_df['Meilleure Permutation (y permuté)'] = perm_comparison_df['Meilleure Permutation (y permuté)'].round(4)


            st.markdown("""
            Le calcul du **R² maximal par permutation** est une analyse complémentaire puissante. Il répond à la question : 
            > "Quel est le meilleur score R² que l'on pourrait obtenir par pur hasard avec ce jeu de données ?"

            Pour le trouver, l'application teste un très grand nombre de permutations de votre variable réponse `y` et retient celle qui donne le R² le plus élevé.

            **Pourquoi est-ce utile ?**

            1.  **Mettre en perspective le R² original** : Un R² de 0.80 peut sembler excellent, mais s'il est possible d'obtenir un R² de 0.95 simplement en mélangeant les données au hasard, alors le score original perd de sa superbe.
            2.  **Détecter le sur-ajustement (Overfitting)** : Si le R² de votre modèle est très proche du R² maximal obtenu par chance, cela peut indiquer que votre modèle est sur-ajusté. Il a peut-être "mémorisé" le bruit dans les données plutôt que d'apprendre une véritable relation sous-jacente.
            3.  **Renforcer la confiance** : Inversement, si le R² de votre modèle est bien inférieur au R² maximal par chance, cela renforce la confiance dans le fait que votre modèle a trouvé une relation authentique et non un artefact statistique.
            """)
            
            st.markdown("---")
            st.markdown("##### Visualisation de la 'Meilleure' Permutation")
            st.markdown("Ci-dessous, la comparaison entre le `y` original et la permutation de `y` qui a produit le R² maximal par chance. **Attention : cette permutation n'a aucune signification scientifique**, elle illustre simplement un artefact statistique.")
            st.dataframe(perm_comparison_df)

        # --- Bouton de téléchargement du rapport ---
        st.markdown("---")
        st.subheader("4. Télécharger le Rapport")
        with st.spinner("Génération du rapport PDF..."):
            pdf_data = create_pdf_report(
                results['df_input'], results['perm_results_df'], results['anova_df'], 
                results['error_metrics_df'], results['residuals_df'], 
                results['summary_df'], results['fig']
            )
        
        st.download_button(
            label="📥 Télécharger le rapport complet (PDF)",
            data=pdf_data,
            file_name="rapport_permutation_test.pdf",
            mime="application/pdf"
        )

        if results['p_value'] >= 0.05:
            st.warning("⚠️ Attention : Le modèle n'est probablement **pas significatif** (le $R^2$ est élevé par le hasard) au seuil de 5%.")
        else:
            st.success("✅ Le modèle est considéré comme **significatif** (faible probabilité d'un $R^2$ dû au hasard) au seuil de 5%.")
        
    st.markdown("---")
    st.caption("Test de Permutation")
