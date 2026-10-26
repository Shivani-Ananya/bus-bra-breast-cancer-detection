-- BreastSight Studio: initial persistent data model.
-- Run this in the Supabase SQL Editor only after reviewing the retention,
-- consent, and de-identification requirements for the intended study.

create extension if not exists pgcrypto;

create table if not exists public.cases (
  id uuid primary key default gen_random_uuid(),
  research_id text not null,
  age numeric check (age is null or (age >= 0 and age <= 120)),
  birads text,
  lesion_size_mm numeric check (lesion_size_mm is null or lesion_size_mm >= 0),
  diagnosis text check (diagnosis is null or diagnosis in ('benign', 'malignant')),
  notes text,
  created_by uuid not null references auth.users(id) on delete cascade default auth.uid(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (created_by, research_id)
);

create table if not exists public.images (
  id uuid primary key default gen_random_uuid(),
  case_id uuid not null references public.cases(id) on delete cascade,
  storage_path text not null unique,
  original_name text not null,
  content_type text,
  size_bytes bigint check (size_bytes is null or size_bytes >= 0),
  created_by uuid not null references auth.users(id) on delete cascade default auth.uid(),
  created_at timestamptz not null default now()
);

create table if not exists public.model_runs (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  model_version text not null,
  config_name text,
  status text not null default 'created'
    check (status in ('created', 'training', 'completed', 'failed', 'archived')),
  metrics jsonb not null default '{}'::jsonb,
  artifact_path text,
  created_by uuid not null references auth.users(id) on delete cascade default auth.uid(),
  created_at timestamptz not null default now(),
  completed_at timestamptz
);

create table if not exists public.predictions (
  id uuid primary key default gen_random_uuid(),
  case_id uuid not null references public.cases(id) on delete cascade,
  model_run_id uuid references public.model_runs(id) on delete set null,
  predicted_class text not null check (predicted_class in ('benign', 'malignant')),
  malignant_probability double precision not null check (malignant_probability between 0 and 1),
  uncertainty double precision check (uncertainty is null or uncertainty between 0 and 1),
  image_uncertainty double precision check (image_uncertainty is null or image_uncertainty between 0 and 1),
  clinical_uncertainty double precision check (clinical_uncertainty is null or clinical_uncertainty between 0 and 1),
  gradcam_path text,
  created_by uuid not null references auth.users(id) on delete cascade default auth.uid(),
  created_at timestamptz not null default now()
);

create table if not exists public.audit_events (
  id bigint generated always as identity primary key,
  actor_id uuid not null references auth.users(id) on delete cascade default auth.uid(),
  action text not null,
  entity_type text not null,
  entity_id text,
  details jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists cases_created_by_idx on public.cases(created_by);
create index if not exists images_case_id_idx on public.images(case_id);
create index if not exists predictions_case_id_idx on public.predictions(case_id);
create index if not exists predictions_model_run_id_idx on public.predictions(model_run_id);
create index if not exists audit_events_actor_id_idx on public.audit_events(actor_id);

alter table public.cases enable row level security;
alter table public.images enable row level security;
alter table public.model_runs enable row level security;
alter table public.predictions enable row level security;
alter table public.audit_events enable row level security;

create policy "users manage their cases" on public.cases
  for all using (created_by = auth.uid()) with check (created_by = auth.uid());
create policy "users manage their images" on public.images
  for all using (created_by = auth.uid()) with check (created_by = auth.uid());
create policy "users manage their model runs" on public.model_runs
  for all using (created_by = auth.uid()) with check (created_by = auth.uid());
create policy "users manage their predictions" on public.predictions
  for all using (created_by = auth.uid()) with check (created_by = auth.uid());
create policy "users read their audit events" on public.audit_events
  for select using (actor_id = auth.uid());
create policy "users create their audit events" on public.audit_events
  for insert with check (actor_id = auth.uid());

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values (
  'breastsight-private',
  'breastsight-private',
  false,
  52428800,
  array['image/png', 'image/jpeg', 'image/webp', 'text/csv',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']
)
on conflict (id) do update set
  public = excluded.public,
  file_size_limit = excluded.file_size_limit,
  allowed_mime_types = excluded.allowed_mime_types;

-- Every object path must begin with the authenticated user's UUID.
create policy "users read their private objects" on storage.objects
  for select to authenticated
  using (bucket_id = 'breastsight-private' and (storage.foldername(name))[1] = auth.uid()::text);
create policy "users upload their private objects" on storage.objects
  for insert to authenticated
  with check (bucket_id = 'breastsight-private' and (storage.foldername(name))[1] = auth.uid()::text);
create policy "users update their private objects" on storage.objects
  for update to authenticated
  using (bucket_id = 'breastsight-private' and (storage.foldername(name))[1] = auth.uid()::text)
  with check (bucket_id = 'breastsight-private' and (storage.foldername(name))[1] = auth.uid()::text);
create policy "users delete their private objects" on storage.objects
  for delete to authenticated
  using (bucket_id = 'breastsight-private' and (storage.foldername(name))[1] = auth.uid()::text);
