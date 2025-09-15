pipeline{
    agent any

    environment{
        VENV_DIR = 'venv'
        GCP_PROJECT = 'mlops-project-471703'
        GCLOUD_PATH = '/var/jenkins_home/google-cloud-sdk/bin'
    }

    stages{

        stage('Cloning github repo to Jenkins'){
            steps{
                script{
                    echo 'Cloning github repo to Jenkins ...'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github_token', url: 'https://github.com/Nikelroid/hotel_cancel_detection-MLOps']])
                }
            }
        }

        stage('Setting up virtual environment and installing dependencies'){
            steps{
                script{
                    echo 'Setting up virtual environment and installing dependencies ...'
                    sh '''
                    python -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    '''
                }
            }
        }
stage('Building and Pushing Docker image to GCR'){
            steps{
                script{
                    withCredentials([file(credentialsId : 'gcp-key' , variable : 'GOOGLE_APPLICATION_CREDENTIALS')]){
                        script{
                            echo 'Building and Pushing Docker image to GCR ...'
                            sh '''
                            set -e  # Exit on any error
                            export PATH=$PATH:${GCLOUD_PATH}
                            
                            # Authenticate and configure GCloud
                            gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                            gcloud config set project ${GCP_PROJECT}
                            gcloud auth configure-docker --quiet
                            
                            # Verify Docker daemon is accessible
                            docker info
                            
                            # Build for Cloud Run compatible platform (amd64)
                            echo "Building Docker image for linux/amd64 platform..."
                            docker build --platform linux/amd64 -t gcr.io/${GCP_PROJECT}/mlops-project:latest .
                            
                            # Push the image
                            echo "Pushing Docker image to GCR..."
                            docker push gcr.io/${GCP_PROJECT}/mlops-project:latest
                            
                            # Verify image architecture
                            echo "Verifying image architecture..."
                            docker manifest inspect gcr.io/${GCP_PROJECT}/mlops-project:latest | grep architecture || true
                            
                            echo "Docker image build and push completed successfully!"
                            '''
                        }
                    }
                }
            }
        }

        stage('Deploy to Google Cloud Run'){
            steps{
                script{
                    withCredentials([file(credentialsId : 'gcp-key' , variable : 'GOOGLE_APPLICATION_CREDENTIALS')]){
                        script{
                            echo 'Deploying to Google Cloud Run...'
                            sh '''
                            set -e  # Exit on any error
                            export PATH=$PATH:${GCLOUD_PATH}
                            
                            # Authenticate and configure GCloud
                            gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                            gcloud config set project ${GCP_PROJECT}
                            
                            # Deploy to Cloud Run with detailed configuration
                            echo "Deploying ml-project to Cloud Run..."
                            gcloud run deploy ml-project \\
                                   --image=gcr.io/${GCP_PROJECT}/mlops-project:latest \\
                                   --platform=managed \\
                                   --region=us-west2 \\
                                   --allow-unauthenticated \\
                                   --port=8080 \\
                                   --memory=2Gi \\
                                   --cpu=1 \\
                                   --min-instances=0 \\
                                   --max-instances=10 \\
                                   --timeout=300 \\
                                   --concurrency=80 \\
                                   --quiet
                            
                            # Get and display the service URL
                            SERVICE_URL=$(gcloud run services describe ml-project --region=us-west2 --format='value(status.url)')
                            echo "Deployment successful!"
                            echo "Service URL: $SERVICE_URL"
                            
                            # Optional: Test the deployment
                            echo "Testing deployment..."
                            curl -s -o /dev/null -w "%{http_code}" $SERVICE_URL || echo "Health check failed, but deployment completed"
                            '''
                        }
                    }
                }
            }
        }

    }
}