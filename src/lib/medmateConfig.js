// MedMate clinical form field definitions — mirrors the Flask backend feature schemas
// (DIABETES_BASE_FEATURES / DEMENTIA_BASE_FEATURES in MedMate_ml.py).

export const DIABETES_GROUPS = [
  {
    title: "Demographics",
    icon: "user",
    fields: [
      {
        name: "age", label: "Age Range", type: "select",
        options: ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"],
        default: "[60-70)",
      },
      {
        name: "gender", label: "Gender", type: "select",
        options: ["Female", "Male"], default: "Female",
      },
    ],
  },
  {
    title: "Hospital Encounter",
    icon: "activity",
    fields: [
      { name: "time_in_hospital", label: "Days in Hospital", type: "number", min: 1, max: 14, default: 4 },
      { name: "num_lab_procedures", label: "Lab Procedures", type: "number", min: 0, max: 132, default: 44 },
      { name: "num_procedures", label: "Procedures", type: "number", min: 0, max: 6, default: 1 },
      { name: "num_medications", label: "Medications", type: "number", min: 0, max: 81, default: 16 },
      { name: "number_diagnoses", label: "Diagnoses", type: "number", min: 1, max: 16, default: 8 },
    ],
  },
  {
    title: "Prior Utilization (past year)",
    icon: "history",
    fields: [
      { name: "number_outpatient", label: "Outpatient Visits", type: "number", min: 0, max: 42, default: 0 },
      { name: "number_emergency", label: "Emergency Visits", type: "number", min: 0, max: 76, default: 0 },
      { name: "number_inpatient", label: "Inpatient Visits", type: "number", min: 0, max: 21, default: 0 },
    ],
  },
  {
    title: "Labs & Medications",
    icon: "flask",
    fields: [
      { name: "A1Cresult", label: "A1C Result", type: "select", options: ["None", "Norm", ">7", ">8"], default: "None" },
      { name: "max_glu_serum", label: "Max Glucose Serum", type: "select", options: ["None", "Norm", ">200", ">300"], default: "None" },
      { name: "insulin", label: "Insulin", type: "select", options: ["No", "Steady", "Up", "Down"], default: "No" },
      { name: "metformin", label: "Metformin", type: "select", options: ["No", "Steady", "Up", "Down"], default: "No" },
      { name: "change", label: "Medication Change", type: "select", options: ["No", "Ch"], default: "No", hint: "Ch = changed" },
      { name: "diabetesMed", label: "On Diabetes Medication", type: "select", options: ["Yes", "No"], default: "Yes" },
    ],
  },
  {
    title: "Diagnosis Codes (ICD-9)",
    icon: "stethoscope",
    fields: [
      { name: "diag_1", label: "Primary Diagnosis", type: "text", default: "250", hint: "e.g. 250 = diabetes" },
      { name: "diag_2", label: "Secondary Diagnosis", type: "text", default: "401" },
      { name: "diag_3", label: "Additional Diagnosis", type: "text", default: "276" },
    ],
  },
];

export const DEMENTIA_GROUPS = [
  {
    title: "Demographics",
    icon: "user",
    fields: [
      { name: "NACCAGE", label: "Age", type: "number", min: 18, max: 110, default: 72 },
      { name: "SEX", label: "Sex", type: "select", options: [{ v: 0, l: "Male" }, { v: 1, l: "Female" }], default: 0 },
      { name: "EDUC", label: "Years of Education", type: "number", min: 0, max: 30, default: 14 },
    ],
  },
  {
    title: "Cognitive Assessment",
    icon: "brain",
    fields: [
      { name: "CDRGLOB", label: "CDR Global", type: "select", options: [{ v: 0.0, l: "0 — None" }, { v: 0.5, l: "0.5 — Questionable" }, { v: 1.0, l: "1 — Mild" }, { v: 2.0, l: "2 — Moderate" }, { v: 3.0, l: "3 — Severe" }], default: 0.0 },
      { name: "CDRSUM", label: "CDR Sum of Boxes", type: "number", min: 0, max: 18, step: 0.5, default: 0 },
      { name: "NACCMMSE", label: "MMSE Score", type: "number", min: 0, max: 30, default: 28, hint: "0–30, lower = worse" },
      { name: "NACCGDS", label: "Geriatric Depression Scale", type: "number", min: 0, max: 15, default: 2 },
      { name: "ANIMALS", label: "Animal Naming (count)", type: "number", min: 0, max: 60, default: 18 },
      { name: "TRAILA", label: "Trail Making A (sec)", type: "number", min: 0, max: 500, default: 35 },
      { name: "TRAILB", label: "Trail Making B (sec)", type: "number", min: 0, max: 500, default: 90 },
    ],
  },
  {
    title: "Clinical & Genetic Markers",
    icon: "heart",
    fields: [
      { name: "DIABETES", label: "Diabetes", type: "select", options: [{ v: 0, l: "No" }, { v: 1, l: "Yes" }], default: 0 },
      { name: "HYPERTEN", label: "Hypertension", type: "select", options: [{ v: 0, l: "No" }, { v: 1, l: "Yes" }], default: 0 },
      { name: "NACCDEP", label: "Depression", type: "select", options: [{ v: 0, l: "No" }, { v: 1, l: "Yes" }], default: 0 },
      { name: "APOE4", label: "APOE4 Carrier", type: "select", options: [{ v: 0, l: "No" }, { v: 1, l: "Yes" }], default: 0 },
      { name: "NACCBMI", label: "BMI", type: "number", min: 10, max: 80, step: 0.1, default: 25.0 },
    ],
  },
];

export function buildDefaults(groups) {
  const out = {};
  groups.forEach((g) => g.fields.forEach((f) => { out[f.name] = f.default; }));
  return out;
}
