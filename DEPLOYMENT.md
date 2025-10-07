# Deployment Guide

## Backend Deployment (Render)

### Step 1: Prepare Repository
```bash
# Make sure all files are committed
git add .
git commit -m "Add FastAPI backend and React frontend"
git push origin main
```

### Step 2: Deploy to Render

1. Go to [https://render.com](https://render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `investment-hypothesis-validator-api`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements-api.txt`
   - **Start Command**: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free

5. Add Environment Variables:
   ```
   AWS_REGION=us-east-1
   AWS_ACCESS_KEY_ID=<your-key>
   AWS_SECRET_ACCESS_KEY=<your-secret>
   OPENSEARCH_URL=<your-opensearch-url>
   OPENSEARCH_INDEX_NAME=investment-documents
   BEDROCK_REGION=us-east-1
   BEDROCK_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0
   BEDROCK_LLM_MODEL=us.anthropic.claude-sonnet-4-20250514-v1:0
   TOP_K_PER_SOURCE=5
   VECTOR_WEIGHT=0.6
   KEYWORD_WEIGHT=0.4
   RECENCY_HALFLIFE_DAYS=14
   ```

6. Click "Create Web Service"

7. Wait for deployment (5-10 minutes)

8. Your API will be live at: `https://investment-hypothesis-validator-api.onrender.com`

### Step 3: Test API
```bash
curl https://your-app.onrender.com/health

curl -X POST https://your-app.onrender.com/validate \
  -H "Content-Type: application/json" \
  -d '{"query": "Is Tesla'\''s gross margin improving?"}'
```

## Frontend Deployment (Vercel)

### Step 1: Prepare Frontend
```bash
cd frontend
npm install
npm run build  # Test build locally
```

### Step 2: Deploy to Vercel

1. Go to [https://vercel.com](https://vercel.com)
2. Click "Add New" → "Project"
3. Import your GitHub repository
4. Configure:
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`

5. Add Environment Variable:
   ```
   VITE_API_URL=https://investment-hypothesis-validator-api.onrender.com
   ```

6. Click "Deploy"

7. Your frontend will be live at: `https://your-app.vercel.app`

## Quick Deployment Commands

### Backend (Render)
```bash
# Render deploys automatically from GitHub
# Just push your changes
git add .
git commit -m "Update backend"
git push origin main
```

### Frontend (Vercel)
```bash
# Install Vercel CLI (optional)
npm i -g vercel

# Deploy from frontend directory
cd frontend
vercel --prod
```

## Testing the Full Stack

1. Open your Vercel URL: `https://your-app.vercel.app`
2. Enter query: "Is Tesla's gross margin improving?"
3. Click "Validate Hypothesis"
4. See results in ~25 seconds

## Troubleshooting

### Backend Issues

**Error: "Module not found"**
- Check `requirements-api.txt` includes all dependencies
- Redeploy on Render

**Error: "AWS credentials not found"**
- Check environment variables in Render dashboard
- Ensure no extra spaces in keys

**Error: "OpenSearch connection failed"**
- Verify `OPENSEARCH_URL` is correct
- Check OpenSearch security group allows Render IPs

### Frontend Issues

**Error: "API request failed"**
- Check `VITE_API_URL` in Vercel environment variables
- Verify backend is running on Render
- Check CORS is enabled in backend

**Error: "Build failed"**
- Run `npm run build` locally to check for errors
- Check `package.json` versions

## Cost Estimate

### Free Tier
- **Render**: Free plan (sleeps after 15 min inactivity)
- **Vercel**: Free plan (100 GB bandwidth/month)
- **Total**: $0/month

### Paid Tier (Recommended for Production)
- **Render**: $7/month (always-on, better performance)
- **Vercel**: Free plan sufficient
- **Total**: $7/month

## Post-Deployment

### Update Backend
```bash
git add .
git commit -m "Update API"
git push origin main
# Render auto-deploys
```

### Update Frontend
```bash
cd frontend
git add .
git commit -m "Update UI"
git push origin main
# Vercel auto-deploys
```

## Demo URLs

After deployment, you'll have:

**Backend API**: https://investment-hypothesis-validator-api.onrender.com
- Health: `/health`
- Docs: `/docs`
- Validate: `/validate` (POST)

**Frontend**: https://investment-hypothesis-validator.vercel.app
- Live demo with UI
- Example queries
- Real-time results

## Interview Demo Strategy

1. **Show live URL** (30 seconds)
   - Open Vercel frontend
   - Show it's actually deployed

2. **Run live query** (30 seconds)
   - Use example: "Is Tesla's gross margin improving?"
   - Show results with citations

3. **Explain architecture** (2 minutes)
   - FastAPI backend on Render
   - React frontend on Vercel
   - Connects to OpenSearch + Bedrock

4. **Show code** (2 minutes)
   - Backend: `src/api/main.py`
   - Frontend: `frontend/src/App.jsx`

Total: ~5 minutes for impressive live demo!
