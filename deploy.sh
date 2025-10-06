#!/bin/bash

echo "🚀 Deploying ExoFindr to Vercel..."

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

# Deploy backend
echo "📡 Deploying Flask backend..."
cd /Users/nachu/space25
vercel --prod

echo "✅ Backend deployed! Note the URL for the next step."

# Deploy frontend
echo "🌐 Deploying Next.js frontend as ExoFindr..."
cd /Users/nachu/space25/nextjs-frontend
vercel --prod --name exofindr

echo "✅ Frontend deployed to https://exofindr.vercel.app"

echo ""
echo "🎉 Deployment complete!"
echo "📝 Don't forget to:"
echo "   1. Set NEXT_PUBLIC_API_URL environment variable in Vercel dashboard"
echo "   2. Upload your model file to the backend"
echo "   3. Test the deployment at https://exofindr.vercel.app"
