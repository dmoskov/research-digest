-- Adds the refusal-tracking columns to items.
--
-- These belong in 001_initial.sql and are now declared there too. This
-- migration exists for databases created by research-digest 0.1.0, whose
-- 001_initial.sql omitted them: those installs recorded 001 as applied, so the
-- corrected 001 will never re-run for them. Idempotent, and a no-op on any
-- database created by 0.1.1 or later.
ALTER TABLE items ADD COLUMN IF NOT EXISTS api_refused_at TIMESTAMPTZ;
ALTER TABLE items ADD COLUMN IF NOT EXISTS api_refusal_type TEXT;
