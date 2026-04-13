"""
CareIQ - Synthetic Data Generator
Génère des données synthétiques réalistes pour le projet CareIQ :
  - Patients
  - Soignants (Caregivers)
  - Signes vitaux (Vital Signs)
  - Avis / Reviews
"""

import os
import random
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker
from faker.providers import BaseProvider

# ─── Configuration ────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

fake_fr = Faker("fr_FR")
fake_ar = Faker("ar_AA")
fake_en = Faker("en_US")
Faker.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../../data/synthetic")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_PATIENTS   = 500
N_CAREGIVERS = 150
N_VITALS_PER_PATIENT = 30   # séries temporelles de 30 mesures par patient
N_REVIEWS    = 800

# ─── Listes de référence ──────────────────────────────────────────────────────
CARE_NEEDS = [
    "nursing",          # soins infirmiers
    "physiotherapy",    # kinésithérapie
    "wound_care",       # soins de plaies
    "medication_admin", # administration médicaments
    "hygiene",          # aide hygiène
    "nutrition",        # aide nutrition
    "mobility",         # aide mobilité
    "palliative",       # soins palliatifs
    "mental_health",    # santé mentale
    "pediatric",        # pédiatrique
]

CAREGIVER_SKILLS = [
    "nursing",
    "physiotherapy",
    "wound_care",
    "medication_admin",
    "hygiene",
    "nutrition",
    "mobility",
    "palliative",
    "mental_health",
    "pediatric",
    "geriatric",        # gériatrie (compétence supplémentaire)
    "diabetes_care",    # diabétologie
]

DIAGNOSES = [
    "Diabète type 2",
    "Hypertension artérielle",
    "Insuffisance cardiaque",
    "BPCO",
    "Maladie d'Alzheimer",
    "AVC séquellaire",
    "Cancer en traitement",
    "Insuffisance rénale chronique",
    "Dépression sévère",
    "Fracture du col du fémur",
    "Démence vasculaire",
    "Parkinson",
    "Diabète type 1",
    "Plaie chronique",
    "Soins palliatifs",
]

LANGUAGES = ["fr", "ar", "fr_ar", "fr_en"]

WILAYAS_ALGERIE = [
    "Alger", "Oran", "Constantine", "Annaba", "Blida",
    "Batna", "Sétif", "Sidi Bel Abbès", "Biskra", "Tébessa",
    "Tlemcen", "Béjaïa", "Tiaret", "Bordj Bou Arréridj", "Médéa",
]

# ─── Provider custom Faker ────────────────────────────────────────────────────
class ClinicalNoteProvider(BaseProvider):
    """Génère des notes cliniques en français/arabe mélangé."""

    FR_TEMPLATES = [
        "Patient présentant {symptom}. Tension artérielle {bp}. Traitement {treatment} en cours.",
        "Visite de contrôle: {symptom}. TA: {bp} mmHg. {observation}.",
        "Renouvellement ordonnance pour {diagnosis}. Patient {status}. {note}.",
        "Urgence à domicile: {symptom}. Appel médecin. {action}.",
        "Bilan mensuel: glycémie {glucose} g/L, TA {bp}. {recommendation}.",
    ]

    SYMPTOMS = [
        "douleurs thoraciques modérées", "fatigue intense", "dyspnée d'effort",
        "oedèmes des membres inférieurs", "confusion temporaire", "chutes répétées",
        "plaie infectée au niveau du pied", "perte d'appétit", "douleurs articulaires",
        "agitation nocturne", "déshydratation légère",
    ]
    TREATMENTS = [
        "Metformine 500mg", "Amlodipine 5mg", "Furosémide 40mg",
        "Ramipril 10mg", "Insuline basale", "Paracétamol 1g",
        "Amoxicilline 1g", "Oméprazole 20mg",
    ]
    OBSERVATIONS = [
        "Etat général stable", "Légère amélioration", "Dégradation progressive",
        "Stable sous traitement", "Surveillance renforcée recommandée",
    ]
    ACTIONS = [
        "Transfert hospitalier envisagé", "Surveillance rapprochée",
        "Pansement réalisé", "Injection effectuée", "Bilan sanguin demandé",
    ]
    RECOMMENDATIONS = [
        "Régime hyposodé conseillé", "Activité physique adaptée",
        "Contrôle glycémique à renforcer", "Consultation spécialisée recommandée",
    ]

    def clinical_note(self) -> str:
        template = random.choice(self.FR_TEMPLATES)
        return template.format(
            symptom=random.choice(self.SYMPTOMS),
            bp=f"{random.randint(110, 180)}/{random.randint(60, 110)}",
            treatment=random.choice(self.TREATMENTS),
            observation=random.choice(self.OBSERVATIONS),
            action=random.choice(self.ACTIONS),
            recommendation=random.choice(self.RECOMMENDATIONS),
            diagnosis=random.choice(DIAGNOSES),
            status=random.choice(["stable", "en amélioration", "en dégradation"]),
            note=fake_fr.sentence(nb_words=8),
            glucose=round(random.uniform(0.7, 3.5), 2),
        )


fake_fr.add_provider(ClinicalNoteProvider)


# ─── 1. Génération Patients ────────────────────────────────────────────────────
def generate_patients(n: int = N_PATIENTS) -> pd.DataFrame:
    """
    Génère n patients synthétiques.
    Colonnes : patient_id, nom, prenom, age, sexe, wilaya, langue,
               diagnostic_principal, nb_besoins, care_needs (liste JSON),
               score_risque_reel, note_clinique, date_creation
    """
    records = []
    for i in range(n):
        age = int(np.random.choice(
            range(18, 95),
            p=_age_distribution(range(18, 95))
        ))
        # Plus le patient est âgé, plus il a de besoins
        nb_needs = min(len(CARE_NEEDS), max(1, int(np.random.poisson(age / 25))))
        needs = random.sample(CARE_NEEDS, nb_needs)

        # Score de risque réel (0-1) basé sur l'âge + nb besoins + diagnostic
        diag = random.choice(DIAGNOSES)
        risk = _compute_risk_score(age, nb_needs, diag)

        records.append({
            "patient_id":          f"PAT_{i+1:04d}",
            "nom":                 fake_fr.last_name(),
            "prenom":              fake_fr.first_name(),
            "age":                 age,
            "sexe":                random.choice(["M", "F"]),
            "wilaya":              random.choice(WILAYAS_ALGERIE),
            "langue":              random.choice(LANGUAGES),
            "diagnostic_principal": diag,
            "nb_besoins":          nb_needs,
            "care_needs":          json.dumps(needs),
            "score_risque_reel":   round(risk, 4),
            "note_clinique":       fake_fr.clinical_note(),
            "date_creation":       fake_fr.date_between(
                                       start_date="-2y", end_date="today"
                                   ).isoformat(),
        })

    df = pd.DataFrame(records)
    path = os.path.join(OUTPUT_DIR, "patients_raw.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[OK] {n} patients -> {path}")
    return df


def _age_distribution(ages):
    """Distribution d'âge réaliste pour soins à domicile (surreprésentation seniors)."""
    probs = []
    for a in ages:
        if a < 40:
            probs.append(0.5)
        elif a < 60:
            probs.append(1.5)
        elif a < 75:
            probs.append(3.0)
        else:
            probs.append(2.0)
    total = sum(probs)
    return [p / total for p in probs]


def _compute_risk_score(age: int, nb_needs: int, diagnosis: str) -> float:
    """Calcule un score de risque patient réaliste (0 à 1)."""
    HIGH_RISK_DIAGS = {
        "Insuffisance cardiaque", "Cancer en traitement",
        "Soins palliatifs", "Maladie d'Alzheimer", "AVC séquellaire",
        "Insuffisance rénale chronique",
    }
    base = (age - 18) / (95 - 18)          # 0-1 selon âge
    need_factor = nb_needs / len(CARE_NEEDS) # 0-1 selon besoins
    diag_factor = 0.3 if diagnosis in HIGH_RISK_DIAGS else 0.0
    noise = np.random.normal(0, 0.05)
    score = 0.4 * base + 0.3 * need_factor + 0.3 * diag_factor + noise
    return float(np.clip(score, 0.0, 1.0))


# ─── 2. Génération Soignants ──────────────────────────────────────────────────
def generate_caregivers(n: int = N_CAREGIVERS) -> pd.DataFrame:
    """
    Génère n soignants synthétiques.
    Colonnes : caregiver_id, nom, prenom, age, sexe, wilaya, langue,
               experience_annees, nb_competences, skills (liste JSON),
               rating_moyen, nb_avis, disponible, tarif_horaire
    """
    records = []
    for i in range(n):
        experience = random.randint(1, 30)
        nb_skills = min(len(CAREGIVER_SKILLS), max(2, int(np.random.poisson(3 + experience / 10))))
        skills = random.sample(CAREGIVER_SKILLS, nb_skills)

        # Rating réaliste : soignants expérimentés mieux notés en moyenne
        base_rating = min(5.0, 3.0 + experience * 0.05 + np.random.normal(0, 0.4))
        nb_avis = max(0, int(np.random.poisson(experience * 4)))

        records.append({
            "caregiver_id":      f"CAR_{i+1:04d}",
            "nom":               fake_fr.last_name(),
            "prenom":            fake_fr.first_name(),
            "age":               random.randint(22, 60),
            "sexe":              random.choice(["M", "F"]),
            "wilaya":            random.choice(WILAYAS_ALGERIE),
            "langue":            random.choice(LANGUAGES),
            "experience_annees": experience,
            "nb_competences":    nb_skills,
            "skills":            json.dumps(skills),
            "rating_moyen":      round(np.clip(base_rating, 1.0, 5.0), 2),
            "nb_avis":           nb_avis,
            "disponible":        random.choices([True, False], weights=[0.7, 0.3])[0],
            "tarif_horaire":     round(random.uniform(500, 3000), 0),  # DZD
        })

    df = pd.DataFrame(records)
    path = os.path.join(OUTPUT_DIR, "caregivers_raw.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[OK] {n} soignants -> {path}")
    return df


# ─── 3. Génération Signes Vitaux ──────────────────────────────────────────────
def generate_vital_signs(patients_df: pd.DataFrame,
                         n_per_patient: int = N_VITALS_PER_PATIENT) -> pd.DataFrame:
    """
    Génère des séries temporelles de signes vitaux pour chaque patient.
    Simule : anomalies (pics, dérives) pour l'entraînement du LSTM Autoencoder.
    Colonnes : vital_id, patient_id, timestamp, frequence_cardiaque,
               pression_systolique, pression_diastolique, temperature,
               saturation_o2, frequence_respiratoire, glycemie, est_anomalie
    """
    records = []
    vital_id = 1

    for _, patient in patients_df.iterrows():
        pid = patient["patient_id"]
        age = patient["age"]
        risk = patient["score_risque_reel"]

        # Baseline personnalisée selon âge et risque
        fc_base   = 70 + age * 0.3 + risk * 20
        sys_base  = 120 + age * 0.5 + risk * 30
        dia_base  = 80 + age * 0.3 + risk * 15
        temp_base = 37.0
        spo2_base = 98 - risk * 5
        rr_base   = 16 + risk * 4
        gluc_base = 1.0 + risk * 1.5

        start_date = datetime.now() - timedelta(days=n_per_patient)

        for j in range(n_per_patient):
            timestamp = start_date + timedelta(hours=j * 8)
            is_anomaly = False

            # Injection d'anomalies (~15% des mesures pour patients à risque élevé)
            anomaly_prob = 0.05 + risk * 0.15
            if random.random() < anomaly_prob:
                is_anomaly = True
                anomaly_type = random.choice(["spike", "drift", "drop"])
                fc_noise   = 30 if anomaly_type == "spike" else -20 if anomaly_type == "drop" else j * 0.5
                sys_noise  = 40 if anomaly_type == "spike" else -25 if anomaly_type == "drop" else j * 0.8
                spo2_noise = -8 if anomaly_type in ("spike", "drop") else -j * 0.2
            else:
                fc_noise   = np.random.normal(0, 5)
                sys_noise  = np.random.normal(0, 8)
                spo2_noise = np.random.normal(0, 1)

            records.append({
                "vital_id":              f"VIT_{vital_id:06d}",
                "patient_id":            pid,
                "timestamp":             timestamp.isoformat(),
                "frequence_cardiaque":   round(np.clip(fc_base + fc_noise, 30, 200), 1),
                "pression_systolique":   round(np.clip(sys_base + sys_noise, 70, 250), 1),
                "pression_diastolique":  round(np.clip(dia_base + np.random.normal(0, 5), 40, 150), 1),
                "temperature":           round(np.clip(temp_base + np.random.normal(0, 0.3) + (1.5 if is_anomaly else 0), 35, 41), 1),
                "saturation_o2":         round(np.clip(spo2_base + spo2_noise, 70, 100), 1),
                "frequence_respiratoire": round(np.clip(rr_base + np.random.normal(0, 2), 8, 40), 1),
                "glycemie":              round(np.clip(gluc_base + np.random.normal(0, 0.2) + (1.5 if is_anomaly else 0), 0.3, 6.0), 2),
                "est_anomalie":          int(is_anomaly),
            })
            vital_id += 1

    df = pd.DataFrame(records)
    path = os.path.join(OUTPUT_DIR, "vital_signs_raw.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[OK] {len(df)} mesures vitaux -> {path}")
    return df


# ─── 4. Génération Reviews / Avis ─────────────────────────────────────────────
def generate_reviews(patients_df: pd.DataFrame,
                     caregivers_df: pd.DataFrame,
                     n: int = N_REVIEWS) -> pd.DataFrame:
    """
    Génère des avis de patients sur les soignants.
    Inclut des faux avis (fake reviews) pour l'entraînement du détecteur.
    Colonnes : review_id, patient_id, caregiver_id, note, texte,
               langue, est_fake, date_avis
    """
    POSITIVE_FR = [
        "Excellent soignant, très professionnel et attentionné.",
        "Très satisfait des soins reçus. Je recommande vivement.",
        "Ponctuel, compétent et très humain. Merci beaucoup.",
        "Prise en charge excellente, mon état s'est nettement amélioré.",
        "Soignant remarquable, à l'écoute et efficace.",
    ]
    NEGATIVE_FR = [
        "Pas ponctuel et peu attentif aux besoins du patient.",
        "Soins insuffisants, je suis très déçu du service.",
        "Manque de professionnalisme, ne recommande pas.",
        "Communication difficile, manque d'empathie.",
        "Plusieurs erreurs lors des soins, inquiétant.",
    ]
    NEUTRAL_FR = [
        "Service correct mais peut mieux faire.",
        "Soins conformes aux attentes, rien d'exceptionnel.",
        "Passable, quelques améliorations seraient souhaitables.",
    ]
    # Faux avis : répétitifs et suspicieusement parfaits
    FAKE_REVIEWS = [
        "PARFAIT !!! Le meilleur soignant de la ville !!! 5 étoiles !!!",
        "Incroyable, absolument parfait, tout le monde devrait faire appel à lui !",
        "Le meilleur, le meilleur, le meilleur. Parfait parfait parfait.",
        "Super super super !!! Vraiment le top du top !!!",
        "Rien à dire, parfait sur tous les points. Note maximale.",
    ]

    patient_ids   = patients_df["patient_id"].tolist()
    caregiver_ids = caregivers_df["caregiver_id"].tolist()
    caregiver_ratings = dict(zip(
        caregivers_df["caregiver_id"],
        caregivers_df["rating_moyen"]
    ))

    records = []
    for i in range(n):
        cid = random.choice(caregiver_ids)
        avg_rating = caregiver_ratings.get(cid, 3.5)
        is_fake = random.random() < 0.12  # 12% de faux avis

        if is_fake:
            note = 5
            texte = random.choice(FAKE_REVIEWS)
            langue = "fr"
        else:
            # Note corrélée au rating moyen du soignant + bruit
            note = int(np.clip(round(avg_rating + np.random.normal(0, 0.8)), 1, 5))
            langue = random.choice(["fr", "fr", "fr", "ar"])  # majorité français
            if note >= 4:
                texte = random.choice(POSITIVE_FR)
            elif note <= 2:
                texte = random.choice(NEGATIVE_FR)
            else:
                texte = random.choice(NEUTRAL_FR)

        records.append({
            "review_id":    f"REV_{i+1:05d}",
            "patient_id":   random.choice(patient_ids),
            "caregiver_id": cid,
            "note":         note,
            "texte":        texte,
            "langue":       langue,
            "est_fake":     int(is_fake),
            "date_avis":    fake_fr.date_between(
                                start_date="-1y", end_date="today"
                            ).isoformat(),
        })

    df = pd.DataFrame(records)
    path = os.path.join(OUTPUT_DIR, "reviews_raw.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[OK] {n} reviews -> {path}")
    return df


# ─── 5. Génération Assignments (Patient ↔ Soignant) ──────────────────────────
def generate_assignments(patients_df: pd.DataFrame,
                         caregivers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Génère les associations patient-soignant (pour le GNN).
    Colonnes : assignment_id, patient_id, caregiver_id,
               match_score, duree_jours, statut, date_debut
    """
    records = []
    assign_id = 1

    for _, patient in patients_df.iterrows():
        patient_needs = set(json.loads(patient["care_needs"]))
        # Choisir entre 1 et 3 soignants par patient
        nb_assignments = random.randint(1, 3)
        assigned = set()

        for _ in range(nb_assignments):
            # Trouver un soignant avec au moins une compétence commune
            attempts = 0
            while attempts < 20:
                cg = caregivers_df.sample(1).iloc[0]
                if cg["caregiver_id"] in assigned:
                    attempts += 1
                    continue
                cg_skills = set(json.loads(cg["skills"]))
                overlap = len(patient_needs & cg_skills)
                if overlap > 0:
                    # Score de matching : overlap + proximité géographique + expérience
                    geo_match = 1.0 if patient["wilaya"] == cg["wilaya"] else 0.3
                    exp_score = min(1.0, cg["experience_annees"] / 15)
                    match = round(0.5 * (overlap / len(patient_needs)) +
                                  0.3 * geo_match + 0.2 * exp_score, 4)
                    records.append({
                        "assignment_id": f"ASS_{assign_id:05d}",
                        "patient_id":    patient["patient_id"],
                        "caregiver_id":  cg["caregiver_id"],
                        "match_score":   match,
                        "duree_jours":   random.randint(7, 365),
                        "statut":        random.choices(
                                             ["actif", "termine", "annule"],
                                             weights=[0.5, 0.4, 0.1]
                                         )[0],
                        "date_debut":    fake_fr.date_between(
                                             start_date="-1y", end_date="today"
                                         ).isoformat(),
                    })
                    assigned.add(cg["caregiver_id"])
                    assign_id += 1
                    break
                attempts += 1

    df = pd.DataFrame(records)
    path = os.path.join(OUTPUT_DIR, "assignments_raw.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[OK] {len(df)} assignments -> {path}")
    return df


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  CareIQ Synthetic Data Generator")
    print(f"  Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\n[1/5] Génération des patients...")
    patients = generate_patients(N_PATIENTS)

    print("\n[2/5] Génération des soignants...")
    caregivers = generate_caregivers(N_CAREGIVERS)

    print("\n[3/5] Génération des signes vitaux...")
    vitals = generate_vital_signs(patients, N_VITALS_PER_PATIENT)

    print("\n[4/5] Génération des reviews...")
    reviews = generate_reviews(patients, caregivers, N_REVIEWS)

    print("\n[5/5] Génération des assignments (graphe patient-soignant)...")
    assignments = generate_assignments(patients, caregivers)

    print("\n" + "=" * 60)
    print("  RESUME DES DONNEES GENEREES")
    print("=" * 60)
    print(f"  Patients    : {len(patients):>6} lignes")
    print(f"  Soignants   : {len(caregivers):>6} lignes")
    print(f"  Signes vitaux: {len(vitals):>5} lignes")
    print(f"  Reviews     : {len(reviews):>6} lignes")
    print(f"  Assignments : {len(assignments):>6} lignes")
    print(f"\n  Fichiers sauvegardes dans : {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)

    # Stats rapides
    print(f"\n  Score risque moyen patients : {patients['score_risque_reel'].mean():.3f}")
    print(f"  % anomalies vitaux          : {vitals['est_anomalie'].mean()*100:.1f}%")
    print(f"  % fake reviews              : {reviews['est_fake'].mean()*100:.1f}%")
    print(f"  Rating moyen soignants      : {caregivers['rating_moyen'].mean():.2f}/5")


if __name__ == "__main__":
    main()
