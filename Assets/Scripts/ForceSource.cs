using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ForceSource : MonoBehaviour
{
    public float magnitude;

    public Vector2 GetForceAt(Vector2 location)
    {
        Vector2 direction = location - new Vector2(transform.position.x, transform.position.y);
        float d = direction.magnitude;
        direction.Normalize();
        return (magnitude * direction) / (d * d);
    }

    public M_ForceSrc GetStruct()
    {
        M_ForceSrc src;
        src.magnitude = magnitude;
        src.pos = transform.position;
        return src;
    }
}
