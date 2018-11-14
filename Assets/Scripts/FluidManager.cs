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

    private float[] densPrev, dens, uPrev, u, vPrev, v;
    private float sampleSize;
    private float sampleHalfSize;

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
    public ComputeShader shaderDef;

    private int kernelComputeForceField;
    private ComputeShader shader;
    private ComputeBuffer _sources;
    private M_ForceSrc[] _sourcesData;

    private ComputeBuffer _forceFieldBuffer;
    private Vector2[] _forceFieldBufferData;

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

        forceSources = FindObjectsOfType<ForceSource>();

        shader = (ComputeShader)Instantiate(shaderDef);
        kernelComputeForceField = shader.FindKernel("CSMain");
        InitShaderData();
    }

    private void InitShaderData()
    {
        shader.SetInt("_N", N);
        shader.SetFloat("_sampleSize", sampleSize);

        _forceFieldBufferData = new Vector2[(N + 2) * (N + 2)];
        _forceFieldBuffer = new ComputeBuffer(_forceFieldBufferData.Length, 2 * 4);
        _forceFieldBuffer.SetData(_forceFieldBufferData);
        shader.SetBuffer(kernelComputeForceField, "Result", _forceFieldBuffer);
    }
    private void UpdateShaderData()
    {
        List<M_ForceSrc> sources = new List<M_ForceSrc>();
        foreach (ForceSource forceSource in forceSources)
        {
            sources.Add(forceSource.GetStruct());
        }
        _sourcesData = sources.ToArray();
        _sources = new ComputeBuffer(_sourcesData.Length, 2 * 4 + 4);
        _sources.SetData(_sourcesData);
        shader.SetBuffer(kernelComputeForceField, "_sources", _sources);
        shader.SetInt("_sourcesCount", _sourcesData.Length);
    }

    private void OnDestroy()
    {
        if (_forceFieldBuffer != null) _forceFieldBuffer.Dispose();
        if (_sources != null) _sources.Dispose();
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
        UpdateShaderData();
        shader.Dispatch(kernelComputeForceField, (N + 2) / 32, (N + 2) / 32, 1);
        _forceFieldBuffer.GetData(_forceFieldBufferData);

        stopwatch.Stop();
        forceFieldIntTime = stopwatch.ElapsedMilliseconds;

        for (int i = 0; i < N + 2; i++)
        {
            for (int j = 0; j < N + 2; j++)
            {
                int index = IX(i, j);
                if (dens[index] > 0f)
                {
                    uPrev[index] += (_forceFieldBufferData[index].x / dens[index]) * Time.deltaTime;
                    vPrev[index] += (_forceFieldBufferData[index].y / dens[index]) * Time.deltaTime;
                }
            }
        }
    }

    private void VelStep(float dt)
    {
        var stopwatch = new System.Diagnostics.Stopwatch();
        stopwatch.Start();
        AddSource(ref u, ref uPrev, dt);
        AddSource(ref v, ref vPrev, dt);
        stopwatch.Stop();
        velAddSourceTime = stopwatch.ElapsedMilliseconds;

        Swap(ref uPrev, ref u);
        Swap(ref vPrev, ref v);

        stopwatch = new System.Diagnostics.Stopwatch();
        stopwatch.Start();
        Diffuse(1, ref u, ref uPrev, visc, dt);
        Diffuse(2, ref v, ref vPrev, visc, dt);
        stopwatch.Stop();
        velDiffuseTime = stopwatch.ElapsedMilliseconds;

        stopwatch = new System.Diagnostics.Stopwatch();
        stopwatch.Start();
        Project(ref u, ref v, ref uPrev, ref vPrev);
        stopwatch.Stop();
        velDiffuseProjTime = stopwatch.ElapsedMilliseconds;

        Swap(ref uPrev, ref u);
        Swap(ref vPrev, ref v);

        stopwatch = new System.Diagnostics.Stopwatch();
        stopwatch.Start();
        Advect(1, ref u, ref uPrev, ref uPrev, ref vPrev, dt);
        Advect(2, ref v, ref vPrev, ref uPrev, ref vPrev, dt);
        stopwatch.Stop();
        velAdvectTime = stopwatch.ElapsedMilliseconds;

        stopwatch = new System.Diagnostics.Stopwatch();
        stopwatch.Start();
        Project(ref u, ref v, ref uPrev, ref vPrev);
        stopwatch.Stop();
        velAdvectProjTime = stopwatch.ElapsedMilliseconds;
    }

    private void DensStep(float dt)
    {
        var stopwatch = new System.Diagnostics.Stopwatch();
        stopwatch.Start();
        AddSource(ref dens, ref densPrev, dt);
        stopwatch.Stop();
        densAddSourceTime = stopwatch.ElapsedMilliseconds;

        Swap(ref densPrev, ref dens);

        stopwatch = new System.Diagnostics.Stopwatch();
        stopwatch.Start();
        Diffuse(0, ref dens, ref densPrev, diff, dt);
        stopwatch.Stop();
        densDiffuseTime = stopwatch.ElapsedMilliseconds;

        Swap(ref densPrev, ref dens);

        stopwatch = new System.Diagnostics.Stopwatch();
        stopwatch.Start();
        Advect(0, ref dens, ref densPrev, ref u, ref v, dt);
        stopwatch.Stop();
        densAdvectTime = stopwatch.ElapsedMilliseconds;
    }

    private void AddSource(ref float[] x, ref float[] s, float dt)
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

    private void Advect (int b, ref float[] d, ref float[] d0, ref float[] u, ref float[] v, float dt)
    {
        int i0, j0, i1, j1;
        float x, y, s0, t0, s1, t1, dt0;
        dt0 = dt * N;
        for (int i = 1; i < N + 1; i++)
        {
            for (int j = 1; j < N + 1; j++)
            {
                x = i - dt0 * u[IX(i, j)]; y = j - dt0 * v[IX(i, j)];
                if (x < 0.5f) x = 0.5f; if (x > N + 0.5f) x = N + 0.5f; i0 = (int)x; i1 = i0 + 1;
                if (y < 0.5f) y = 0.5f; if (y > N + 0.5) y = N + 0.5f; j0 = (int)y; j1 = j0 + 1;
                s1 = x - i0; s0 = 1f - s1; t1 = y - j0; t0 = 1f - t1;
                d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)])+
                            s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
            }
        }
        SetBnd(b, ref d);
    }

    private void Project (ref float[] u, ref float[] v, ref float[] p, ref float[] div)
    {
        float h;
        h = 1f / N;
        for (int i = 1; i < N + 1; i++)
        {
            for (int j = 1; j < N + 1; j++)
            {
                div[IX(i, j)] = -0.5f * h * (u[IX(i + 1, j)] - u[IX(i - 1, j)] +
                v[IX(i, j + 1)] - v[IX(i, j - 1)]);
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
                    p[IX(i, j)] = (div[IX(i, j)] + p[IX(i - 1, j)] + p[IX(i + 1, j)] +
                     p[IX(i, j - 1)] + p[IX(i, j + 1)]) / 4;
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

        int i = 1;
        int j = 1;
        float val;
        for(float x = sampleHalfSize; x < 1; x += sampleSize)
        {
            j = 1;
            for (float y = sampleHalfSize; y < 1; y+= sampleSize)
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
