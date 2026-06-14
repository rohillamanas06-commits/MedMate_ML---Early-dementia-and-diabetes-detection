import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

// Renders a single clinical form field (select / number / text).
export default function Field({ field, value, onChange }) {
  const id = `field-${field.name}`;
  const testId = `input-${field.name}`;

  const isObjOptions = field.type === "select" && typeof field.options?.[0] === "object";

  return (
    <div className="flex flex-col gap-1.5">
      <Label htmlFor={id} className="text-[13px] font-medium text-foreground/80">
        {field.label}<span className="text-red-500 ml-1">*</span>
      </Label>

      {field.type === "select" ? (
        <Select
          required
          value={String(value)}
          onValueChange={(v) => onChange(field.name, isObjOptions ? coerce(v) : v)}
        >
          <SelectTrigger id={id} data-testid={testId} className="rounded-xl bg-secondary/40 border-border focus:ring-primary">
            <SelectValue placeholder="Select" />
          </SelectTrigger>
          <SelectContent>
            {field.options.map((opt) => {
              const v = isObjOptions ? opt.v : opt;
              const l = isObjOptions ? opt.l : opt;
              return (
                <SelectItem key={String(v)} value={String(v)} data-testid={`${testId}-opt-${String(v)}`}>
                  {l}
                </SelectItem>
              );
            })}
          </SelectContent>
        </Select>
      ) : (
        <Input
          required
          id={id}
          data-testid={testId}
          type={field.type === "number" ? "number" : "text"}
          inputMode={field.type === "number" ? "decimal" : "text"}
          min={field.min}
          max={field.max}
          step={field.step}
          value={value ?? ""}
          onChange={(e) => onChange(field.name, field.type === "number" ? toNum(e.target.value) : e.target.value)}
          className="rounded-xl bg-secondary/40 border-border focus-visible:ring-primary"
        />
      )}

      {field.hint && <span className="text-[11px] text-muted-foreground">{field.hint}</span>}
    </div>
  );
}

function coerce(v) {
  const n = Number(v);
  return Number.isNaN(n) ? v : n;
}
function toNum(v) {
  if (v === "") return "";
  const n = Number(v);
  return Number.isNaN(n) ? v : n;
}
