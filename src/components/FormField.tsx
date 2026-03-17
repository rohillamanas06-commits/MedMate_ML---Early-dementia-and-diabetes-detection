import { ReactNode } from "react";

interface FormFieldProps {
  label: string;
  description?: string;
  children: ReactNode;
}

export const FormField = ({ label, description, children }: FormFieldProps) => (
  <div className="space-y-1.5">
    <label className="text-sm font-medium text-foreground">{label}</label>
    {description && <p className="text-xs text-muted-foreground leading-snug">{description}</p>}
    {children}
  </div>
);

interface ToggleOptionProps {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
}

export const ToggleOption = ({ label, value, onChange }: ToggleOptionProps) => (
  <button
    type="button"
    onClick={() => onChange(!value)}
    className={`flex items-center justify-between px-3 sm:px-4 py-2.5 sm:py-3 rounded-xl transition-all duration-200 ${
      value
        ? "bg-primary text-primary-foreground shadow-layered"
        : "bg-muted text-muted-foreground hover:bg-accent"
    }`}
  >
    <span className="text-xs sm:text-sm font-medium text-left pr-2">{label}</span>
    <span className="text-xs font-mono shrink-0">{value ? "Yes" : "No"}</span>
  </button>
);