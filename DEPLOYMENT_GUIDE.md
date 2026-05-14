# 🚀 Render.com Deployment Guide

This guide walks you through deploying the Crop Recommendation System on Render.com for **FREE**.

---

## ✅ Pre-Deployment Checklist

- [x] Code pushed to GitHub: `https://github.com/Jittub45/Hybrid-Uncertainty-Aware-Model`
- [x] `runtime.txt` created (Python 3.11.3)
- [x] `Procfile` configured for Gunicorn
- [x] `.env.example` updated with all required vars
- [x] Gemini API key obtained
- [x] `psycopg2-binary` in `requirements.txt`

---

## 📋 Step 1: Sign Up on Render.com

1. Go to **https://render.com**
2. Click **Sign Up**
3. Use your email: `jitendrakumaryadav2003@gmail.com`
4. Verify email
5. **Connect GitHub account** (click "Connect GitHub" in the dashboard)
   - Authorize Render to access your GitHub repos
   - Select `Jittub45/Hybrid-Uncertainty-Aware-Model` repo

---

## 🔧 Step 2: Create a New Web Service on Render

1. In Render dashboard, click **+ New**
2. Select **Web Service**
3. Choose repo: **Hybrid-Uncertainty-Aware-Model**
4. Fill in details:
   - **Name**: `crop-recommender` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --chdir app flask_app:app --bind 0.0.0.0:$PORT`
   - **Plan**: Select **Free** tier

---

## 🗄️ Step 3: Set Up PostgreSQL Database

1. In Render dashboard, click **+ New**
2. Select **PostgreSQL**
3. Configure:
   - **Name**: `crop-db`
   - **Database**: `cropdb`
   - **User**: `cropuser`
   - **Region**: Same as Web Service
   - **Plan**: **Free** tier
   
4. After creation, **copy the Internal Database URL** (looks like):
   ```
   postgresql://cropuser:password@localhost:5432/cropdb
   ```
   ⚠️ **DO NOT** use this! Use the **External Database URL** instead:
   ```
   postgresql://cropuser:password@host.render.internal:5432/cropdb
   ```

---

## 🔐 Step 4: Configure Environment Variables

On Render Web Service page, scroll to **Environment**:

Add these environment variables:

| Key | Value | Source |
|-----|-------|--------|
| `DATABASE_URL` | `postgresql://user:pass@host:5432/cropdb` | From PostgreSQL database page |
| `GEMINI_API_KEY` | `AIzaSyBLCq-JAjb56qpZa3PI98qAlCMQ8OuS93Q` | You provided this |
| `SECRET_KEY` | Generate a random 32+ char string | Use: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `PORT` | `10000` | Auto-set by Render |
| `SMTP_HOST` | `smtp.gmail.com` | Your SMTP (optional) |
| `SMTP_PORT` | `587` | Your SMTP (optional) |
| `SMTP_USER` | `your_email@gmail.com` | Your SMTP (optional) |
| `SMTP_PASSWORD` | `your_app_password` | Your SMTP (optional) |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Default |

**For dev mode (no email):**
- Leave `SMTP_*` variables empty → OTP will print to logs

---

## 🚀 Step 5: Deploy

1. Click **Create Web Service**
2. Render will:
   - ✅ Clone your GitHub repo
   - ✅ Install dependencies from `requirements.txt`
   - ✅ Create PostgreSQL tables automatically
   - ✅ Start the server

**⏳ First deployment: 3-5 minutes**

Once complete, your app is live at:
```
https://crop-recommender.onrender.com
```

---

## ✅ Step 6: Verify Deployment

1. Open https://crop-recommender.onrender.com
2. Test each feature:
   - [ ] Signup page loads
   - [ ] Can sign up with new user
   - [ ] OTP verification works (check Render logs)
   - [ ] Can log in
   - [ ] Crop prediction works
   - [ ] Chatbot responds
   - [ ] Schemes API returns data
   - [ ] Insights page shows charts

---

## 📊 Monitoring & Logs

To view logs on Render:

1. Go to Web Service dashboard
2. Click **Logs** tab
3. Watch for errors in real-time

**Common issues:**
- `UnicodeEncodeError`: Already fixed (UTF-8 set)
- `ModuleNotFoundError`: Missing dependency in `requirements.txt`
- `OperationalError`: Database connection failed (check `DATABASE_URL`)

---

## 🛠️ Troubleshooting

### App won't start?
- Check logs: Render > Web Service > Logs
- Common fix: Restart deployment (`Manual Deploy` > `Deploy latest`)

### Database connection fails?
- Verify `DATABASE_URL` matches PostgreSQL page exactly
- Make sure PostgreSQL is in same region

### Gemini chatbot returns error?
- Check if `GEMINI_API_KEY` is set correctly
- Verify API key hasn't expired at https://ai.google.dev/

### OTP not sending (expected in dev mode)?
- Check Render logs for dev OTP code
- Look for line: `[DEV OTP] recipient@email.com -> 123456`

---

## 💾 Database Management

### View/manage database on Render:

1. Go to PostgreSQL instance
2. Click **Connect**
3. Use **psql** connection string in terminal:
   ```bash
   psql postgresql://user:pass@host:5432/cropdb
   ```

### Clear all data (start fresh):

```sql
DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS otp_challenges CASCADE;
```

Then restart Web Service to recreate tables.

---

## 🔄 Redeploy After Changes

After making code changes locally:

```bash
git add .
git commit -m "your message"
git push origin master
```

Render auto-detects push → auto-deploys in ~2 minutes.

---

## 📞 Support

- **Render Docs**: https://render.com/docs
- **Flask Docs**: https://flask.palletsprojects.com/
- **PostgreSQL**: https://www.postgresql.org/docs/
- **Gemini API**: https://ai.google.dev/

---

## ✨ Success Indicators

After deployment, you should see:

1. ✅ App loads at https://crop-recommender.onrender.com
2. ✅ Database: Users saved in PostgreSQL
3. ✅ Authentication: Signup/login works
4. ✅ ML Model: Predictions return in <500ms
5. ✅ Chatbot: Gemini AI responds
6. ✅ Schemes: API returns agriculture schemes
7. ✅ Insights: Shows model performance metrics

**Enjoy your live crop recommendation system! 🌾**
