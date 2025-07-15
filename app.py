from fastapi import FastAPI, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, Column, String, DateTime, JSON, Text, Boolean, ForeignKey, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import asyncio
import aiohttp
import json
import uuid
import os
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from contextlib import asynccontextmanager
import networkx as nx
from enum import Enum

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/workflow_db")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    workflows = relationship("Workflow", back_populates="owner")

class Workflow(Base):
    __tablename__ = "workflows"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, index=True)
    description = Column(Text)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    data = Column(JSON)  # Store workflow structure
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    owner = relationship("User", back_populates="workflows")
    executions = relationship("WorkflowExecution", back_populates="workflow")

class WorkflowExecution(Base):
    __tablename__ = "workflow_executions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_id = Column(UUID(as_uuid=True), ForeignKey("workflows.id"))
    status = Column(String, default="pending")  # pending, running, completed, failed
    input_data = Column(JSON)
    output_data = Column(JSON)
    logs = Column(JSON)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    error_message = Column(Text)
    
    workflow = relationship("Workflow", back_populates="executions")

# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    is_active: bool
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str

class WorkflowCreate(BaseModel):
    name: str
    description: Optional[str] = None
    data: Dict[str, Any]
    is_public: bool = False

class WorkflowResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    data: Dict[str, Any]
    is_public: bool
    created_at: datetime
    updated_at: datetime

class BlockType(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    LLM = "llm"
    TRANSFORM = "transform"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    HTTP = "http"
    DATABASE = "database"
    FILE = "file"
    MATH = "math"
    TEXT = "text"
    DELAY = "delay"

class BlockConfig(BaseModel):
    type: BlockType
    position: Dict[str, float]
    config: Dict[str, Any]

class WorkflowExecutionRequest(BaseModel):
    blocks: List[BlockConfig]
    connections: List[Dict[str, str]]
    input_data: Optional[Dict[str, Any]] = None

# Block execution engine
class BlockExecutor:
    def __init__(self):
        self.executors = {
            BlockType.INPUT: self._execute_input,
            BlockType.OUTPUT: self._execute_output,
            BlockType.LLM: self._execute_llm,
            BlockType.TRANSFORM: self._execute_transform,
            BlockType.CONDITIONAL: self._execute_conditional,
            BlockType.LOOP: self._execute_loop,
            BlockType.HTTP: self._execute_http,
            BlockType.DATABASE: self._execute_database,
            BlockType.FILE: self._execute_file,
            BlockType.MATH: self._execute_math,
            BlockType.TEXT: self._execute_text,
            BlockType.DELAY: self._execute_delay,
        }
    
    async def execute_block(self, block: BlockConfig, inputs: Dict[str, Any]) -> Dict[str, Any]:
        executor = self.executors.get(block.type)
        if not executor:
            raise ValueError(f"Unknown block type: {block.type}")
        return await executor(block, inputs)
    
    async def _execute_input(self, block: BlockConfig, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"output": block.config.get("text", "")}
    
    async def _execute_output(self, block: BlockConfig, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {"output": inputs.get("input", "")}
    
    async def _execute_llm(self, block: BlockConfig, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model = block.config.get("model", "llama2")
        prompt_template = block.config.get("prompt", "{input}")
        temperature = block.config.get("temperature", 0.7)
        
        # Replace placeholders in prompt
        prompt = prompt_template.format(**inputs)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"output": result.get("response", "")}
                else:
                    error_text = await response.text()
                    raise HTTPException(status_code=500, detail=f"Ollama error: {error_text}")
    
    async def _execute_transform(self, block: BlockConfig, inputs: Dict[str, Any]) -> Dict[str, Any]:
        transform_type = block.config.get("type", "uppercase")
        input_text = inputs.get("input", "")
        
        if transform_type == "uppercase":
            return {"output": input_text.upper()}
        elif transform_type == "lowercase":
            return {"output": input_text.lower()}
        elif transform_type == "trim":
            return {"output": input_text.strip()}
        elif transform_type == "reverse":
            return {"output": input_text[::-1]}
        else:
            return {"output": input_text}
    
    async def _execute_conditional(self, block: BlockConfig, inputs: Dict[str, Any]) -> Dict[str, Any]:
        condition = block.config.get("condition", "")
        value = inputs.get("input", "")
        
        try:
            # Simple condition evaluation (can be expanded)
            if condition == "not_empty":
                result = bool(value and str(value).strip())
            elif condition == "is_number":
                result = str(value).isdigit()
            elif condition.startswith("contains:"):
                search_term = condition.split(":", 1)[1]
                result = search_term in str(value)
            else:
                result = bool(value)
            
            return {"output": result, "true_output": value if result else "", "false_output": value if not result else ""}
        except Exception:
            return {"output": False, "true_output": "", "false_output": value}
    
    async def _execute_loop(self, block: BlockConfig, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Simple loop implementation
        iterations = block.config.get("iterations", 1)
        input_value = inputs.get("input", "")
        results = []
        
        for i in range(iterations):
            results.append(f"{input_value} (iteration {i+1})")
        
        return {"output": results}
    
    async def _execute_http(self, block: BlockConfig, inputs: Dict[str, Any]) -> Dict[str, Any]:
        method = block.config.get("method", "GET")
        url = block.config.get("url", "")
        headers = block.config.get("headers", {})
        
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, headers=headers) as response:
                text = await response.text()
                return {"output": text, "status_code": response.status}
    
    async def _execute_database(self, block: BlockConfig, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Placeholder for database operations
        return {"output": "Database operation completed"}
    
    async def _execute_file(self, block: BlockConfig, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Placeholder for file operations
        return {"output": "File operation completed"}
    
    async def _execute_math(self, block: BlockConfig, inputs: Dict[str, Any]) -> Dict[str, Any]:
        operation = block.config.get("operation", "add")
        value1 = float(inputs.get("input1", 0))
        value2 = float(inputs.get("input2", 0))
        
        if operation == "add":
            result = value1 + value2
        elif operation == "subtract":
            result = value1 - value2
        elif operation == "multiply":
            result = value1 * value2
        elif operation == "divide":
            result = value1 / value2 if value2 != 0 else 0
        else:
            result = 0
        
        return {"output": result}
    
    async def _execute_text(self, block: BlockConfig, inputs: Dict[str, Any]) -> Dict[str, Any]:
        operation = block.config.get("operation", "concat")
        
        if operation == "concat":
            text1 = str(inputs.get("input1", ""))
            text2 = str(inputs.get("input2", ""))
            return {"output": text1 + text2}
        elif operation == "split":
            text = str(inputs.get("input", ""))
            delimiter = block.config.get("delimiter", " ")
            return {"output": text.split(delimiter)}
        else:
            return {"output": str(inputs.get("input", ""))}
    
    async def _execute_delay(self, block: BlockConfig, inputs: Dict[str, Any]) -> Dict[str, Any]:
        delay_seconds = block.config.get("seconds", 1)
        await asyncio.sleep(delay_seconds)
        return {"output": inputs.get("input", "")}

# Workflow execution engine
class WorkflowEngine:
    def __init__(self):
        self.executor = BlockExecutor()
    
    async def execute_workflow(self, blocks: List[BlockConfig], connections: List[Dict[str, str]], 
                             input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Build execution graph
        graph = nx.DiGraph()
        
        # Add nodes
        block_map = {block.config.get("id", str(uuid.uuid4())): block for block in blocks}
        for block_id, block in block_map.items():
            graph.add_node(block_id, block=block)
        
        # Add edges
        for connection in connections:
            graph.add_edge(connection["from"], connection["to"])
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Workflow contains cycles")
        
        # Execute in topological order
        execution_order = list(nx.topological_sort(graph))
        results = {}
        
        for block_id in execution_order:
            block = block_map[block_id]
            
            # Gather inputs from predecessor blocks
            inputs = {}
            for pred_id in graph.predecessors(block_id):
                if pred_id in results:
                    inputs.update(results[pred_id])
            
            # Add initial input data for input blocks
            if block.type == BlockType.INPUT and input_data:
                inputs.update(input_data)
            
            # Execute block
            result = await self.executor.execute_block(block, inputs)
            results[block_id] = result
        
        return results

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# Application setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create database tables
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(
    title="LLM Workflow Builder",
    description="A modern workflow builder for LLM applications",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize workflow engine
workflow_engine = WorkflowEngine()

# Routes
@app.post("/auth/register", response_model=UserResponse)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return UserResponse(
        id=str(db_user.id),
        username=db_user.username,
        email=db_user.email,
        is_active=db_user.is_active,
        created_at=db_user.created_at
    )

@app.post("/auth/login", response_model=Token)
async def login(username: str, password: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/workflows", response_model=List[WorkflowResponse])
async def get_workflows(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    workflows = db.query(Workflow).filter(Workflow.owner_id == current_user.id).all()
    return [
        WorkflowResponse(
            id=str(w.id),
            name=w.name,
            description=w.description,
            data=w.data,
            is_public=w.is_public,
            created_at=w.created_at,
            updated_at=w.updated_at
        )
        for w in workflows
    ]

@app.post("/workflows", response_model=WorkflowResponse)
async def create_workflow(
    workflow: WorkflowCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    db_workflow = Workflow(
        name=workflow.name,
        description=workflow.description,
        data=workflow.data,
        is_public=workflow.is_public,
        owner_id=current_user.id
    )
    db.add(db_workflow)
    db.commit()
    db.refresh(db_workflow)
    
    return WorkflowResponse(
        id=str(db_workflow.id),
        name=db_workflow.name,
        description=db_workflow.description,
        data=db_workflow.data,
        is_public=db_workflow.is_public,
        created_at=db_workflow.created_at,
        updated_at=db_workflow.updated_at
    )

@app.post("/workflows/execute")
async def execute_workflow(
    execution_request: WorkflowExecutionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Create execution record
        execution = WorkflowExecution(
            workflow_id=None,  # For ad-hoc executions
            status="running",
            input_data=execution_request.input_data or {}
        )
        db.add(execution)
        db.commit()
        db.refresh(execution)
        
        # Execute workflow
        results = await workflow_engine.execute_workflow(
            execution_request.blocks,
            execution_request.connections,
            execution_request.input_data
        )
        
        # Update execution record
        execution.status = "completed"
        execution.output_data = results
        execution.completed_at = datetime.utcnow()
        db.commit()
        
        return {
            "success": True,
            "execution_id": str(execution.id),
            "results": results
        }
    
    except Exception as e:
        # Update execution record with error
        execution.status = "failed"
        execution.error_message = str(e)
        execution.completed_at = datetime.utcnow()
        db.commit()
        
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/block-types")
async def get_block_types():
    return {
        "input": {
            "name": "Text Input",
            "description": "Provides text input to the workflow",
            "config": {
                "text": {"type": "textarea", "label": "Input Text", "default": ""}
            }
        },
        "output": {
            "name": "Output Display",
            "description": "Displays workflow output",
            "config": {}
        },
        "llm": {
            "name": "LLM Query",
            "description": "Query a Large Language Model",
            "config": {
                "model": {"type": "select", "label": "Model", "options": ["llama2", "mistral", "codellama"], "default": "llama2"},
                "prompt": {"type": "textarea", "label": "Prompt Template", "default": "Answer the following: {input}"},
                "temperature": {"type": "number", "label": "Temperature", "min": 0, "max": 2, "step": 0.1, "default": 0.7}
            }
        },
        "transform": {
            "name": "Text Transform",
            "description": "Transform text data",
            "config": {
                "type": {"type": "select", "label": "Transform Type", "options": ["uppercase", "lowercase", "trim", "reverse"], "default": "uppercase"}
            }
        },
        "conditional": {
            "name": "Conditional Logic",
            "description": "Conditional branching based on input",
            "config": {
                "condition": {"type": "select", "label": "Condition", "options": ["not_empty", "is_number", "contains:"], "default": "not_empty"}
            }
        },
        "math": {
            "name": "Math Operation",
            "description": "Perform mathematical operations",
            "config": {
                "operation": {"type": "select", "label": "Operation", "options": ["add", "subtract", "multiply", "divide"], "default": "add"}
            }
        },
        "text": {
            "name": "Text Operation",
            "description": "Text manipulation operations",
            "config": {
                "operation": {"type": "select", "label": "Operation", "options": ["concat", "split"], "default": "concat"},
                "delimiter": {"type": "text", "label": "Delimiter", "default": " "}
            }
        },
        "delay": {
            "name": "Delay",
            "description": "Add delay to workflow execution",
            "config": {
                "seconds": {"type": "number", "label": "Delay (seconds)", "min": 0, "default": 1}
            }
        },
        "http": {
            "name": "HTTP Request",
            "description": "Make HTTP requests",
            "config": {
                "method": {"type": "select", "label": "Method", "options": ["GET", "POST", "PUT", "DELETE"], "default": "GET"},
                "url": {"type": "text", "label": "URL", "default": ""},
                "headers": {"type": "json", "label": "Headers", "default": {}}
            }
        }
    }

@app.websocket("/ws/workflow/{workflow_id}")
async def websocket_endpoint(websocket: WebSocket, workflow_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Handle real-time workflow updates
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
