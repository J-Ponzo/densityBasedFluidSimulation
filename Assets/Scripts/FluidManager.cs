using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FluidManager : MonoBehaviour
{
    //FLUID SIM
    public int N = 1024;
    public float visc = 0.5f;
    public float diff = 0.5f;
    public float densPerSec = 100f;
    public float speedLim = 10f;

    private float[] densPrev, dens, uPrev, u, vPrev, v, p, div;
    private float sampleSize;
    private float sampleHalfSize;

    public int maxSource = 32;
    private ForceSource[] forceSources;

    //LOGS
    private long forceFieldIntTime;
    private long densAddSourceTime;
    private long densDiffuseTime;
    private long densAdvectTime;
    private long velAddSourceTime;
    private long velDiffuseTime;
    private long velDiffuseProjTime;
    private long velAdvectTime;
    private long velAdvectProjTime;

    //COMPUTE SHADERS
    //Force Field
    public ComputeShader shaderForceFieldDef;

    private int kernelComputeForceField;
    private ComputeShader shaderForceField;
    private ComputeBuffer _sourcesBuffer;
    private M_ForceSrc[] _sourcesData;

    private ComputeBuffer _forceFieldBuffer;
    private Vector2[] _forceFieldBufferData;

    //Diffusion
    public ComputeShader shaderDiffusionDef;

    private int kernelDiffusion;
    private ComputeShader shaderDiffusion;
    private ComputeBuffer _x0Buffer;
    private ComputeBuffer _xBuffer;

    //Advection
    public ComputeShader shaderAdvectionDef;

    private int kernelAdvection;
    private ComputeShader shaderAdvection;
    private ComputeBuffer _dBuffer;
    private ComputeBuffer _d0Buffer;
    private ComputeBuffer _uaBuffer;
    private ComputeBuffer _vaBuffer;

    //Projection Compute
    public ComputeShader shaderProjectionComputeDef;

    private int kernelProjectionCompute;
    private ComputeShader shaderProjectionCompute;
    private ComputeBuffer _uInBuffer;
    private ComputeBuffer _vInBuffer;
    private ComputeBuffer _pOutBuffer;

    //Projection Apply
    public ComputeShader shaderProjectionApplyDef;

    private int kernelProjectionApply;
    private ComputeShader shaderProjectionApply;
    private ComputeBuffer _uOutBuffer;
    private ComputeBuffer _vOutBuffer;
    private ComputeBuffer _pInBuffer;

    // Start is called before the first frame update
    void Start()
    {
        sampleSize = 1f / (float)N;
        sampleHalfSize = sampleSize / 2f;

        densPrev = new float[(N + 2) * (N + 2)];
        dens = new float[(N + 2) * (N + 2)];
        uPrev = new float[(N + 2) * (N + 2)];
        u = new float[(N + 2) * (N + 2)];
        vPrev = new float[(N + 2) * (N + 2)];
        v = new float[(N + 2) * (N + 2)];
        p = new float[(N + 2) * (N + 2)];
        div = new float[(N + 2) * (N + 2)];

        forceSources = FindObjectsOfType<ForceSource>();

        shaderForceField = (ComputeShader)Instantiate(shaderForceFieldDef);
        kernelComputeForceField = shaderForceField.FindKernel("CSMain");
        shaderDiffusion = (ComputeShader)Instantiate(shaderDiffusionDef);
        kernelDiffusion = shaderForceField.FindKernel("CSMain");
        shaderAdvection = (ComputeShader)Instantiate(shaderAdvectionDef);
        kernelAdvection = shaderForceField.FindKernel("CSMain");
        shaderProjectionCompute = (ComputeShader)Instantiate(shaderProjectionComputeDef);
        kernelProjectionCompute = shaderProjectionCompute.FindKernel("CSMain");
        shaderProjectionApply = (ComputeShader)Instantiate(shaderProjectionApplyDef);
        kernelProjectionApply = shaderProjectionApply.FindKernel("CSMain");

        InitShaderData();
    }

    private void InitShaderData()
    {
        //Force field shader init
        shaderForceField.SetInt("_N", N);
        _forceFieldBufferData = new Vector2[(N + 2) * (N + 2)];
        _forceFieldBuffer = new ComputeBuffer(_forceFieldBufferData.Length, 2 * 4);
        _forceFieldBuffer.SetData(_forceFieldBufferData);
        shaderForceField.SetBuffer(kernelComputeForceField, "Result", _forceFieldBuffer);
        _sourcesBuffer = new ComputeBuffer(maxSource, 2 * 4 + 4);
        shaderForceField.SetBuffer(kernelComputeForceField, "_sources", _sourcesBuffer);

        //Diffusion shader init
        shaderDiffusion.SetInt("_N", N);
        _x0Buffer = new ComputeBuffer((N + 2) * (N + 2), 4);
        shaderDiffusion.SetBuffer(kernelDiffusion, "_x0", _x0Buffer);
        _xBuffer = new ComputeBuffer((N + 2) * (N + 2), 4);
        shaderDiffusion.SetBuffer(kernelDiffusion, "Result", _xBuffer);

        //Advection shader init
        shaderAdvection.SetInt("_N", N);
        _dBuffer = new ComputeBuffer((N + 2) * (N + 2), 4);
        shaderAdvection.SetBuffer(kernelAdvection, "_d", _dBuffer);
        _d0Buffer = new ComputeBuffer((N + 2) * (N + 2), 4);
        shaderAdvection.SetBuffer(kernelAdvection, "_d0", _d0Buffer);
        _uaBuffer = new ComputeBuffer((N + 2) * (N + 2), 4);
        shaderAdvection.SetBuffer(kernelAdvection, "_u", _uaBuffer);
        _vaBuffer = new ComputeBuffer((N + 2) * (N + 2), 4);
        shaderAdvection.SetBuffer(kernelAdvection, "_v", _vaBuffer);

        //ProjectionCompute shader init
        shaderProjectionCompute.SetInt("_N", N);
        _uInBuffer = new ComputeBuffer((N + 2) * (N + 2), 4);
        shaderProjectionCompute.SetBuffer(kernelProjectionCompute, "_u", _uInBuffer);
        _vInBuffer = new ComputeBuffer((N + 2) * (N + 2), 4);
        shaderProjectionCompute.SetBuffer(kernelProjectionCompute, "_v", _vInBuffer);
        _pOutBuffer = new ComputeBuffer((N + 2) * (N + 2), 4);
        shaderProjectionCompute.SetBuffer(kernelProjectionCompute, "_p", _pOutBuffer);

        //ProjectionApply shader init
        shaderProjectionApply.SetInt("_N", N);
        _uOutBuffer = new ComputeBuffer((N + 2) * (N + 2), 4);
        shaderProjectionApply.SetBuffer(kernelProjectionApply, "_u", _uOutBuffer);
        _vOutBuffer = new ComputeBuffer((N + 2) * (N + 2), 4);
        shaderProjectionApply.SetBuffer(kernelProjectionApply, "_v", _vOutBuffer);
        _pInBuffer = new ComputeBuffer((N + 2) * (N + 2), 4);
        shaderProjectionApply.SetBuffer(kernelProjectionApply, "_p", _pInBuffer);
    }
    private void UpdateForceFieldShaderData()
    {
        int nbSourcesSent = Math.Min(forceSources.Length, maxSource);
        shaderForceField.SetInt("_sourcesCount", nbSourcesSent);

        _sourcesData = new M_ForceSrc[maxSource];
        for (int i = 0; i < nbSourcesSent; i++) {
            _sourcesData[i] = forceSources[i].GetStruct();
        }
        _sourcesBuffer.SetData(_sourcesData);
    }

    private void UpdateDiffusionShaderData(int b, float[] x, float[] x0, float diff, float dt)
    {
        shaderDiffusion.SetInt("_b", b);
        shaderDiffusion.SetFloat("_diff", diff);
        shaderDiffusion.SetFloat("_dt", dt);

        _x0Buffer.SetData(x0);
        _xBuffer.SetData(x);
    }

    private void UpdateAdvectionShaderData(int b, float[] d, float[] d0, float[] u, float[] v, float dt)
    {
        shaderAdvection.SetInt("_b", b);
        shaderAdvection.SetFloat("_dt", dt);

        _dBuffer.SetData(d);
        _d0Buffer.SetData(d0);
        _uaBuffer.SetData(u);
        _vaBuffer.SetData(v);
    }

    private void UpdateProjectionComputeShaderData(float[] u, float[] v, float[] p, float[] div)
    {
        _uInBuffer.SetData(u);
        _vInBuffer.SetData(v);
        _pOutBuffer.SetData(p);
    }

    private void UpdateProjectionApplyShaderData(float[] u, float[] v, float[] p, float[] div)
    {
        _uOutBuffer.SetData(u);
        _vOutBuffer.SetData(v);
        _pInBuffer.SetData(p);
    }

    private void OnDestroy()
    {
        if (_sourcesBuffer != null) _sourcesBuffer.Dispose();
        if (_forceFieldBuffer != null) _forceFieldBuffer.Dispose();
        if (_x0Buffer != null) _x0Buffer.Dispose();
        if (_xBuffer != null) _xBuffer.Dispose();
        if (_dBuffer != null) _dBuffer.Dispose();
        if (_d0Buffer != null) _d0Buffer.Dispose();
        if (_uaBuffer != null) _uaBuffer.Dispose();
        if (_vaBuffer != null) _vaBuffer.Dispose();
        if (_uInBuffer != null) _uInBuffer.Dispose();
        if (_vInBuffer != null) _vInBuffer.Dispose();
        if (_pOutBuffer != null) _pOutBuffer.Dispose();
        if (_uOutBuffer != null) _uOutBuffer.Dispose();
        if (_vOutBuffer != null) _vOutBuffer.Dispose();
        if (_pInBuffer != null) _pInBuffer.Dispose();
    }

    int IX(int i, int j)
    {
        return i + (N + 2) * j;
    }

    int IX(float u, float v)
    {
        int i = (int) ((float) u / sampleSize) + 1;
        int j = (int)((float) v / sampleSize) + 1; ;
        return IX(i, j);
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        float dt = Time.deltaTime;
        GetFroUI();
        VelStep(dt);
        DensStep(dt);
        DrawStep();
    }

    private void GetFroUI()
    {
        if (Input.GetKey(KeyCode.Mouse1))
        {
            densPrev[IX(Input.mousePosition.x / (float) Screen.width, Input.mousePosition.y / (float) Screen.height)] = densPerSec * Time.deltaTime;
        }

        var stopwatch = new System.Diagnostics.Stopwatch();
        stopwatch.Start();

        // Update shader's dynamic data 
        UpdateForceFieldShaderData();
        shaderForceField.Dispatch(kernelComputeForceField, (N + 2) / 16, (N + 2) / 16, 1);
        _forceFieldBuffer.GetData(_forceFieldBufferData);

        for (int i = 0; i < N + 2; i++)
        {
            for (int j = 0; j < N + 2; j++)
            {
                int index = IX(i, j);
                if (Mathf.Abs(dens[index]) > 0f)
                {
                    uPrev[index] += (_forceFieldBufferData[index].x / dens[index]) * Time.deltaTime;
                    vPrev[index] += (_forceFieldBufferData[index].y / dens[index]) * Time.deltaTime;
                    uPrev[index] = Mathf.Clamp(uPrev[index], -speedLim, speedLim);
                    vPrev[index] = Mathf.Clamp(vPrev[index], -speedLim, speedLim);
                }
            }
        }

        stopwatch.Stop();
        forceFieldIntTime = stopwatch.ElapsedMilliseconds;
    }

    private void VelStep(float dt)
    {
        var stopwatch = new System.Diagnostics.Stopwatch();
        stopwatch.Start();
        AddSource(ref u, uPrev, dt);
        AddSource(ref v, vPrev, dt);
        stopwatch.Stop();
        velAddSourceTime = stopwatch.ElapsedMilliseconds;

        Swap(ref uPrev, ref u);
        Swap(ref vPrev, ref v);

        stopwatch = new System.Diagnostics.Stopwatch();
        stopwatch.Start();


        UpdateDiffusionShaderData(1, u, uPrev, visc, dt);
        for (int i = 0; i < 20; i++)
        {
            shaderDiffusion.Dispatch(kernelDiffusion, (N + 2) / 32, (N + 2) / 32, 1);
        }
        _xBuffer.GetData(u);
        //Diffuse(1, ref u, ref uPrev, visc, dt);

        UpdateDiffusionShaderData(2, v, vPrev, visc, dt);
        for (int i = 0; i < 20; i++)
        {
            shaderDiffusion.Dispatch(kernelDiffusion, (N + 2) / 32, (N + 2) / 32, 1);
        }
        _xBuffer.GetData(v);
        //Diffuse(2, ref v, ref vPrev, visc, dt);


        stopwatch.Stop();
        velDiffuseTime = stopwatch.ElapsedMilliseconds;

        stopwatch = new System.Diagnostics.Stopwatch();
        stopwatch.Start();


        UpdateProjectionComputeShaderData(u, v, p, div);
        for (int i = 0; i < 20; i++)
        {
            shaderProjectionCompute.Dispatch(kernelProjectionCompute, (N + 2) / 16, (N + 2) / 16, 1);
        }
        _pOutBuffer.GetData(p);
        UpdateProjectionApplyShaderData(u, v, p, div);
        shaderProjectionApply.Dispatch(kernelProjectionApply, (N + 2) / 16, (N + 2) / 16, 1);
        _uOutBuffer.GetData(u);
        _vOutBuffer.GetData(v);
        //Project(ref u, ref v, uPrev, vPrev);


        stopwatch.Stop();
        velDiffuseProjTime = stopwatch.ElapsedMilliseconds;

        Swap(ref uPrev, ref u);
        Swap(ref vPrev, ref v);

        stopwatch = new System.Diagnostics.Stopwatch();
        stopwatch.Start();

        UpdateAdvectionShaderData(1, u, uPrev, uPrev, vPrev, dt);
        shaderAdvection.Dispatch(kernelAdvection, (N + 2) / 32, (N + 2) / 32, 1);
        _dBuffer.GetData(u);
        //Advect(1, ref u, uPrev, uPrev, vPrev, dt);

        UpdateAdvectionShaderData(2, v, vPrev, uPrev, vPrev, dt);
        shaderAdvection.Dispatch(kernelAdvection, (N + 2) / 32, (N + 2) / 32, 1);
        _dBuffer.GetData(v);
        //Advect(2, ref v, vPrev, uPrev, vPrev, dt);


        stopwatch.Stop();
        velAdvectTime = stopwatch.ElapsedMilliseconds;

        stopwatch = new System.Diagnostics.Stopwatch();
        stopwatch.Start();


        UpdateProjectionComputeShaderData(u, v, p, div);
        for (int i = 0; i < 20; i++)
        {
            shaderProjectionCompute.Dispatch(kernelProjectionCompute, (N + 2) / 16, (N + 2) / 16, 1);
        }
        _pOutBuffer.GetData(p);
        UpdateProjectionApplyShaderData(u, v, p, div);
        shaderProjectionApply.Dispatch(kernelProjectionApply, (N + 2) / 16, (N + 2) / 16, 1);
        _uOutBuffer.GetData(u);
        _vOutBuffer.GetData(v);
        //Project(ref u, ref v, uPrev, vPrev);


        stopwatch.Stop();
        velAdvectProjTime = stopwatch.ElapsedMilliseconds;
    }

    private void DensStep(float dt)
    {
        var stopwatch = new System.Diagnostics.Stopwatch();
        stopwatch.Start();
        AddSource(ref dens, densPrev, dt);
        stopwatch.Stop();
        densAddSourceTime = stopwatch.ElapsedMilliseconds;

        Swap(ref densPrev, ref dens);

        stopwatch = new System.Diagnostics.Stopwatch();
        stopwatch.Start();

        UpdateDiffusionShaderData(0, dens, densPrev, diff, dt);
        for (int i = 0; i < 20; i++)
        {
            shaderDiffusion.Dispatch(kernelDiffusion, (N + 2) / 32, (N + 2) / 32, 1);
        }
        _xBuffer.GetData(dens);
        //Diffuse(0, ref dens, ref densPrev, diff, dt);


        stopwatch.Stop();
        densDiffuseTime = stopwatch.ElapsedMilliseconds;

        Swap(ref densPrev, ref dens);

        stopwatch = new System.Diagnostics.Stopwatch();
        stopwatch.Start();


        UpdateAdvectionShaderData(0, dens, densPrev, u, v, dt);
        shaderAdvection.Dispatch(kernelAdvection, (N + 2) / 32, (N + 2) / 32, 1);
        _dBuffer.GetData(dens);
        //Advect(0, ref dens, densPrev, u, v, dt);


        stopwatch.Stop();
        densAdvectTime = stopwatch.ElapsedMilliseconds;
    }

    private void AddSource(ref float[] x, float[] s, float dt)
    {
        for (int i = 0;  i < x.Length; i++)
        {
            x[i] += dt * s[i];
        }
    }

    private void Diffuse(int b, ref float[] x, ref float[] x0, float diff, float dt)
    {
        float a = dt * diff * N * N;
        for (int k = 0; k < 20; k++)
        {
            for (int i = 1; i < N + 1; i++)
            {
                for (int j = 1; j < N + 1; j++)
                {
                    x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] + x[IX(i, j - 1)] + x[IX(i, j + 1)])) / (1f + 4f * a);
                }
            }
            SetBnd(b, ref x);
        }
    }

    private void Advect (int b, ref float[] d, float[] d0, float[] u, float[] v, float dt)
    {
        int i0, j0, i1, j1;
        float x, y, s0, t0, s1, t1, dt0;
        dt0 = dt * N;
        for (int i = 1; i < N + 1; i++)
        {
            for (int j = 1; j < N + 1; j++)
            {
                x = i - dt0 * u[IX(i, j)]; y = j - dt0 * v[IX(i, j)];
                if (x < 0.5f) x = 0.5f;if (x > N + 0.5f) x = N + 0.5f;
                i0 = (int)x; i1 = i0 + 1;
                if (y < 0.5f) y = 0.5f; if (y > N + 0.5) y = N + 0.5f; j0 = (int)y; j1 = j0 + 1;
                s1 = x - i0; s0 = 1f - s1; t1 = y - j0; t0 = 1f - t1;
                d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                            s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
            }
        }
        SetBnd(b, ref d);
    }

    private void Project (ref float[] u, ref float[] v, float[] p, float[] div)
    {
        float h;
        h = 1f / N;
        for (int i = 1; i < N + 1; i++)
        {
            for (int j = 1; j < N + 1; j++)
            {
                div[IX(i, j)] = -0.5f * h * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]);
                p[IX(i, j)] = 0;
            }
        }
        SetBnd(0, ref div); SetBnd(0, ref p);
        for (int k = 0; k < 20; k++)
        {
            for (int i = 1; i < N + 1; i++)
            {
                for (int j = 1; j < N + 1; j++)
                {
                    p[IX(i, j)] = (div[IX(i, j)] + p[IX(i - 1, j)] + p[IX(i + 1, j)] + p[IX(i, j - 1)] + p[IX(i, j + 1)]) / 4f;
                }
            }
            SetBnd(0, ref p);
        }
        for (int i = 1; i < N + 1; i++)
        {
            for (int j = 1; j < N + 1; j++)
            {
                u[IX(i, j)] -= 0.5f * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) / h;
                v[IX(i, j)] -= 0.5f * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) / h;
            }
        }
        SetBnd(1, ref u); SetBnd(2, ref v);
    }

    private void Swap(ref float[] x0, ref float[] x)
    {
        float[] tmp = x0;
        x0 = x;
        x = tmp;
    }

    private void SetBnd(int b, ref float[] x)
    {
        for (int i = 1; i < N + 1; i++)
        {
            x[IX(0, i)] = b == 1 ? -x[IX(1, i)] : x[IX(1, i)];
            x[IX(N + 1, i)] = b == 1 ? -x[IX(N, i)] : x[IX(N, i)];
            x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
            x[IX(i, N + 1)] = b == 2 ? -x[IX(i, N)] : x
                [IX(i, N)];
        }
        x[IX(0, 0)] = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
        x[IX(0, N + 1)] = 0.5f * (x[IX(1, N + 1)] + x[IX(0, N)]);
        x[IX(N + 1, 0)] = 0.5f * (x[IX(N, 0)] + x[IX(N + 1, 1)]);
        x[IX(N + 1, N + 1)] = 0.5f * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);
    }

    private void DrawStep()
    {
        //TODO implement shading
    }

    void OnDrawGizmos()
    {
        if (sampleSize == 0)
        {
            return;
        }

        int i = 0;
        int j = 0;
        float val;
        for (float x = sampleHalfSize; x < 1; x += sampleSize)
        {
            j = 0;
            for (float y = sampleHalfSize; y < 1; y += sampleSize)
            {
                val = dens[IX(i, j)];
                Gizmos.color = new Color(val, val, val, 1f);
                //Gizmos.color = new Color(((forceField[IX(i, j)].x / 1000f) + 1f) / 2f, ((forceField[IX(i, j)].y / 1000f) + 1f) / 2f, 0f, 1f);
                //Gizmos.color = new Color(((u[IX(i, j)]) + 1f) / 2f, ((v[IX(i, j)]) + 1f) / 2f, 0f, 1f);
                Gizmos.DrawCube(new Vector3(x, y, 0f), new Vector3(sampleSize, sampleSize, 1f));
                j++;
            }
            i++;
        }
    }

    private void OnGUI()
    {
        GUIStyle statStyle = new GUIStyle();
        statStyle.fontSize = 30;
        statStyle.normal.textColor = Color.white;

        float frameTime = forceFieldIntTime + densAddSourceTime + densDiffuseTime + densAdvectTime + velAddSourceTime 
            + velDiffuseTime + velDiffuseProjTime + velAdvectTime + velAdvectProjTime;
        float time = (float)(frameTime) / 1000f;
        float fps = (float)Math.Round(1f / time, 2);
        GUI.Label(new Rect(10, 10, 100, 50), "fps : " + fps, statStyle);
        GUI.Label(new Rect(170, 10, 100, 50), "(" + time * 1000f + " ms)", statStyle);

        time = (float)(forceFieldIntTime) / 1000f;
        GUI.Label(new Rect(10, 40, 100, 50), "forces field (" + time * 1000f + " ms)", statStyle);

        time = (float)(densAddSourceTime) / 1000f;
        GUI.Label(new Rect(10, 70, 100, 50), "dens. src (" + time * 1000f + " ms)", statStyle);

        time = (float)(densDiffuseTime) / 1000f;
        GUI.Label(new Rect(10, 100, 100, 50), "dens. diff. (" + time * 1000f + " ms)", statStyle);

        time = (float)(densAdvectTime) / 1000f;
        GUI.Label(new Rect(10, 130, 100, 50), "dens. adv. (" + time * 1000f + " ms)", statStyle);

        time = (float)(velAddSourceTime) / 1000f;
        GUI.Label(new Rect(10, 160, 100, 50), "vel. src  (" + time * 1000f + " ms)", statStyle);

        time = (float)(velDiffuseTime) / 1000f;
        GUI.Label(new Rect(10, 190, 100, 50), "vel. diff. (" + time * 1000f + " ms)", statStyle);

        time = (float)(velDiffuseProjTime) / 1000f;
        GUI.Label(new Rect(10, 220, 100, 50), "vel. diff. proj. (" + time * 1000f + " ms)", statStyle);

        time = (float)(velAdvectTime) / 1000f;
        GUI.Label(new Rect(10, 250, 100, 50), "vel. adv. (" + time * 1000f + " ms)", statStyle);

        time = (float)(velAdvectProjTime) / 1000f;
        GUI.Label(new Rect(10, 280, 100, 50), "vel. adv. proj. (" + time * 1000f + " ms)", statStyle);
    }
}
