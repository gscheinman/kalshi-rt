# External cron setup for reliable snapshots

GitHub Actions silently skips cron triggers on low-activity workflows. We saw
~17 runs/24h when expecting 96. Two-layer defense:

1. The workflow's own crons now use off-peak minute offsets (:03, :08, :13,
   :18, :23, :28, :33, :38, :43, :48, :53, :58 -- avoiding the :00 boundary).
2. An external scheduler hits the GitHub REST API every 15 min. This bypasses
   GitHub's own scheduler entirely.

## One-time setup (5 minutes)

### Step 1 -- Create a fine-grained Personal Access Token

1. Go to https://github.com/settings/personal-access-tokens/new
2. **Token name:** `kalshi-rt external snapshot trigger`
3. **Expiration:** 1 year (or whatever feels right)
4. **Repository access:** Only select `gscheinman/kalshi-rt`
5. **Permissions** under "Repository permissions":
   - **Contents:** Read and write  (required for the `/dispatches` endpoint)
   - **Metadata:** Read (auto-required)

   NOTE: "Actions: write" is NOT what the dispatches endpoint needs.
   It's the slightly counterintuitive Contents permission. If you
   already created the token with the wrong permission, you can edit
   it in place at https://github.com/settings/personal-access-tokens
   without regenerating the token string.
6. Click **Generate token**, copy it. You will NOT be able to see it again.

### Step 2 -- Set up cron-job.org (free)

1. Go to https://cron-job.org and create an account (free tier supports
   every-1-min jobs, way more than we need).
2. Click **Create cronjob**.
3. **Title:** `Kalshi RT snapshot trigger`
4. **URL:** `https://api.github.com/repos/gscheinman/kalshi-rt/dispatches`
5. **Schedule:** Every 15 minutes.
6. Under **Advanced**:
   - **Request method:** POST
   - **Request headers:**
     ```
     Accept: application/vnd.github+json
     Authorization: Bearer YOUR_TOKEN_FROM_STEP_1
     X-GitHub-Api-Version: 2022-11-28
     ```
   - **Request body:**
     ```json
     {"event_type": "trigger-snapshot"}
     ```
7. Save. Click **Run now** to verify.

### Step 3 -- Verify it works

Within ~30 seconds, the snapshot workflow should appear at
https://github.com/gscheinman/kalshi-rt/actions/workflows/snapshot.yml
with trigger type `repository_dispatch`.

If you see it, you're done. If not, check the cron-job.org execution log --
it'll show the HTTP response from GitHub.

## What you get

- Reliable snapshots every ~15 min regardless of GitHub's cron skipping
- The existing weekly settlement trigger from snapshot.yml continues to work
- If cron-job.org goes down, the GitHub-internal cron still fires (as well as
  it ever did)
- If GitHub Actions is unhealthy, both will fail safely

## Operational notes

- Concurrency group at the job level dedups overlapping runs, so it's safe
  if both schedulers fire close together.
- The repository_dispatch event_type is `trigger-snapshot`. If you ever
  need a different external trigger (e.g. for a different workflow), use a
  different event_type and add a matching entry under
  `on.repository_dispatch.types`.

## Cost

- cron-job.org free tier: every 1 min, unlimited jobs
- GitHub Actions on this public repo: unlimited minutes
- Total cost: $0/month
