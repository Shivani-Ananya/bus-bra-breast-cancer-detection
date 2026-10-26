# Supabase setup

## 1. Create the project

Create a private Supabase project and select the appropriate data region. Do not upload research data until the study's de-identification, consent, retention, and access requirements have been approved.

## 2. Apply the schema

Open **SQL Editor** in the Supabase dashboard, copy the complete contents of `migrations/0001_initial.sql`, review it, and run it once.

The migration creates:

- `cases`
- `images`
- `model_runs`
- `predictions`
- `audit_events`
- Private `breastsight-private` Storage bucket
- Row-Level Security policies
- Per-user Storage policies

Object paths must begin with the authenticated user's UUID:

```text
USER_UUID/cases/CASE_UUID/scan.png
```

## 3. Configure authentication

Enable only the sign-in methods approved for the project. For a research prototype, disable unrestricted public sign-up unless it is explicitly required. Use invited accounts or an approved organization identity provider where possible.

## 4. Configure local secrets

Copy `.env.example` to `.env.local` and replace the placeholders. Never commit `.env.local`.

```env
SUPABASE_URL=https://nglqmgxgryctgjrywsnt.supabase.co
SUPABASE_ANON_KEY=your-publishable-key
SUPABASE_SERVICE_ROLE_KEY=your-secret-service-role-key
```

The ordinary `/api/cases` endpoints use the caller's Supabase access token plus the public key, allowing Row-Level Security to authorize every request. The service-role key is reserved for future trusted server-side maintenance and must never be returned to the browser.

## 5. Configure Vercel

Add the same three variables in **Vercel → Project → Settings → Environment Variables**, then redeploy. Keep the service-role value available only to the server runtime.

## 6. Verify

Before credentials are added:

```text
GET /api/supabase/status
```

returns `configured: false`. After configuration and deployment it returns `configured: true` without revealing any credentials.

Authenticated case requests require:

```http
Authorization: Bearer SUPABASE_USER_ACCESS_TOKEN
```

Available endpoints:

```text
GET  /api/cases
POST /api/cases
```

## Retention and backups

Live rows and Storage objects remain until deleted or until the Supabase project is removed. Database-backup recovery windows depend on the selected Supabase plan. Database backups do not restore deleted Storage objects, so define a separate backup and retention policy for ultrasound files before production use.
